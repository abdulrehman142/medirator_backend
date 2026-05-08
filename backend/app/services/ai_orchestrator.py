import asyncio
from concurrent.futures import TimeoutError as FuturesTimeoutError
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.gemini import GeminiService
from app.services.xrayas_service import xrayas_service_instance

class AIOrchestrator:
    def __init__(self):
        self.xrayas = xrayas_service_instance
        self.settings = get_settings()
        self.gemini_timeout_seconds = 45.0
        self.xrayas_timeout_seconds = 45.0
        self.gemini = GeminiService(self.settings)

    def _safe_answer(self) -> str:
        return (
            "I am unable to provide a reliable automated answer right now. "
            "Please try again in a moment and consult a licensed clinician for urgent concerns."
        )

    def _safety_fallback_message(self, gemini_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        base = self._safe_answer()
        hints: List[str] = []
        if not self.gemini.configured():
            hints.append(
                "Add GEMINI_API_KEY to backend/.env (see Google AI Studio) and restart the API — Gemini handles chat and file-mode responses."
            )
        elif gemini_result and not gemini_result.get("ok"):
            err = (gemini_result.get("error") or "").lower()
            if "timeout" in err:
                hints.append("Gemini timed out; check network latency or retry.")
            elif "429" in err or "resource exhausted" in err or "rate" in err:
                hints.append(
                    "Gemini returned HTTP 429 (rate limit / quota). Wait a minute and retry, or enable billing."
                )
            elif "quota" in err or "billing" in err or "credit" in err:
                hints.append("Gemini request may be hitting quota/billing limits on your Google Cloud / AI Studio project.")
            elif "permission" in err or "403" in err:
                hints.append("Gemini returned permission denied — confirm the API key is valid for the Generative Language API.")
            else:
                hints.append("Gemini failed after retries — verify the API key, model availability, and project settings.")

        if not hints:
            return {"answer": base, "confidence": 0.0, "model_used": "System Safety Fallback"}

        return {
            "answer": base + "\n\n" + "\n".join(f"- {hint}" for hint in hints),
            "confidence": 0.0,
            "model_used": "System Safety Fallback",
        }

    def _is_app_question(self, text: str) -> bool:
        keywords = [
            "app",
            "route",
            "page",
            "patient",
            "doctor",
            "medibot",
            "login",
            "register",
            "appointments",
            "profile",
            "family-history",
            "family history",
            "salts",
            "health-risks",
            "report-analysis",
            "visualizer",
            "doctor/pages",
            "home",
            "about",
            "faqs",
            "privacy",
            "terms",
            "admin",
            "dashboard",
            "frontend",
            "backend",
        ]
        lower_text = text.lower()
        return any(keyword in lower_text for keyword in keywords)

    def _format_conversation_context(self, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
        # Build conversation context
        context = ""
        if conversation_history:
            context = "Previous conversation:\n"
            for msg in conversation_history:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                context += f"{role}: {content}\n"
            context += "\n"
        return context

    def _format_file_context(self, file_contexts: Optional[List[Dict[str, Any]]] = None) -> str:
        if not file_contexts:
            return ""
        lines = ["Uploaded file context (treat this as primary evidence when relevant):"]
        for item in file_contexts:
            name = item.get("filename", "unknown_file")
            extracted = item.get("extracted_text", "")
            if extracted:
                lines.append(f"- {name}: {extracted}")
        return "\n".join(lines) + "\n\n"

    def _build_gemini_prompt(
        self,
        text: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        file_contexts: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        context = self._format_conversation_context(conversation_history)
        file_context = self._format_file_context(file_contexts)

        if self._is_app_question(text):
            return (
                "You are the MediRator chatbot assistant. Use the app overview below when answering frontend or app-related questions. "
                "Act as a medical chatbot for the app, not as Gemini, and keep responses concise.\n\n"
                f"{APP_OVERVIEW}\n\n"
                f"{context}"
                f"{file_context}"
                f"User question: {text}"
            )

        return (
            "You are Gemini, a general-purpose conversational assistant. "
            "Answer clearly and helpfully, prioritize uploaded file evidence when available, and make medical explanations easy to understand. "
            "If the question is not medical, answer in a friendly, concise way.\n\n"
            f"{context}"
            f"{file_context}"
            f"User question: {text}"
        )

    async def _call_xrayas(self, image_path: str, text: str) -> Dict[str, Any]:
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(self.xrayas.analyze, image_path, text),
                timeout=self.xrayas_timeout_seconds,
            )
            answer = (response.get("answer", "") or "").strip()
            if not response.get("ok") or not answer:
                return {
                    "ok": False,
                    "answer": "",
                    "confidence": 0.0,
                    "error": "xrayas_failed",
                }
            return {
                "ok": True,
                "answer": answer,
                "confidence": float(response.get("confidence", 0.0) or 0.0),
                "error": None,
                "model": response.get("model_name") or "XRayAS",
            }
        except (asyncio.TimeoutError, FuturesTimeoutError):
            return {"ok": False, "answer": "", "confidence": 0.0, "error": "xrayas_timeout"}
        except Exception as e:
            return {"ok": False, "answer": "", "confidence": 0.0, "error": f"xrayas_error:{e}"}

    async def _call_gemini(self, prompt: str) -> Dict[str, Any]:
        if not self.gemini.configured():
            return {"ok": False, "answer": "", "confidence": 0.0, "error": "gemini_unavailable", "model": None}

        fallback_model_name = self.gemini.get_supported_model() or "models/gemini-2.0-flash"
        try:
            answer, model_name = await asyncio.wait_for(
                asyncio.to_thread(self.gemini.generate_sync, prompt),
                timeout=self.gemini_timeout_seconds,
            )
            if not answer:
                raise ValueError("gemini_empty_response")
            return {
                "ok": True,
                "answer": answer,
                "confidence": 0.92,
                "error": None,
                "model": model_name,
            }
        except (asyncio.TimeoutError, FuturesTimeoutError):
            return {"ok": False, "answer": "", "confidence": 0.0, "error": "gemini_timeout", "model": fallback_model_name}
        except Exception as e:
            return {"ok": False, "answer": "", "confidence": 0.0, "error": f"gemini_error:{e}", "model": fallback_model_name}

    def _build_exact_result(
        self,
        *,
        answer: str,
        confidence: float,
        model_used: str,
        actual_model_used: str,
        selection_reason: str,
    ) -> Dict[str, Any]:
        return {
            "answer": answer,
            "confidence": confidence,
            "model_used": model_used,
            "actual_model_used": actual_model_used,
            "selection_reason": selection_reason,
        }

    async def get_runtime_status(self) -> Dict[str, Any]:
        gemini_model = None
        gemini_error = None
        gemini_ready = self.gemini.configured()

        if self.gemini.configured():
            try:
                gemini_model = self.gemini.get_supported_model()
            except Exception as e:
                gemini_error = str(e)
                gemini_model = None

        xrayas_ready = self.xrayas.ready
        if not xrayas_ready:
            try:
                await asyncio.to_thread(self.xrayas.load_model)
                xrayas_ready = self.xrayas.ready
            except Exception:
                xrayas_ready = False

        return {
            "gemini": {
                "enabled": self.gemini.configured(),
                "ready": gemini_ready,
                "model": gemini_model,
                "error": gemini_error,
            },
            "xrayas": {
                "enabled": True,
                "ready": xrayas_ready,
                "load_error": getattr(self.xrayas, "load_error", None),
                "model_path": getattr(self.xrayas, "model_path", None),
            },
        }

    def _collect_file_context_from_history(
        self,
        conversation_history: Optional[List[Dict[str, Any]]],
        current_files: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        contexts: List[Dict[str, Any]] = []
        for item in current_files or []:
            if item.get("extracted_text"):
                contexts.append(item)

        for msg in conversation_history or []:
            for file_info in msg.get("files", []) or []:
                extracted = file_info.get("extracted_text")
                if extracted:
                    contexts.append(
                        {
                            "filename": file_info.get("filename", "unknown_file"),
                            "content_type": file_info.get("content_type", "unknown"),
                            "extracted_text": extracted,
                        }
                    )
        # de-dupe by filename + text prefix
        deduped: List[Dict[str, Any]] = []
        seen = set()
        for item in contexts:
            key = f"{item.get('filename','')}::{(item.get('extracted_text','') or '')[:200]}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    async def process_interaction(
        self,
        text: str,
        image_path: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        uploaded_file_contexts: Optional[List[Dict[str, Any]]] = None,
        model_preference: str = "gemini",
    ) -> Dict[str, Any]:
        """
        Coordinates the pipeline for Gemini and XRayAS.
        Accepts conversation history for context-aware Gemini responses.
        """
        file_contexts = self._collect_file_context_from_history(
            conversation_history=conversation_history,
            current_files=uploaded_file_contexts,
        )

        disclaimer = "This is not a medical diagnosis. Please consult a professional for medical advice."
        gemini_prompt = self._build_gemini_prompt(
            text=text,
            conversation_history=conversation_history,
            file_contexts=file_contexts,
        )

        pref = (model_preference or "gemini").lower()
        has_uploaded_files = bool(image_path or uploaded_file_contexts)

        if pref == "xrayas":
            if not image_path:
                raise RuntimeError("XRayAS requires an uploaded chest x-ray image. Please upload a valid x-ray image file.")

            xray_result = await self._call_xrayas(image_path=image_path, text=text)
            if not xray_result or not xray_result.get("ok") or not xray_result.get("answer"):
                raise RuntimeError(f"XRayAS failed: {(xray_result or {}).get('error') or 'unknown_error'}")

            return {
                "answer": xray_result.get("answer", ""),
                "confidence": float(xray_result.get("confidence", 0.0)),
                "model_used": "XRayAS",
                "actual_model_used": str(xray_result.get("model") or "XRayAS"),
                "selection_reason": "User selected XRayAS for chest x-ray image analysis.",
                "disclaimer": "This is an automated chest x-ray screening prediction. Consult a radiologist for clinical confirmation.",
            }

        if has_uploaded_files:
            gemini_result = await self._call_gemini(gemini_prompt)
            if not gemini_result or not gemini_result.get("ok") or not gemini_result.get("answer"):
                raise RuntimeError(f"Gemini failed for file mode: {(gemini_result or {}).get('error') or 'unknown_error'}")

            return {
                "answer": gemini_result.get("answer", ""),
                "confidence": float(gemini_result.get("confidence", 0.0)),
                "model_used": "Gemini",
                "actual_model_used": str(gemini_result.get("model") or "Gemini"),
                "selection_reason": "Gemini handled the uploaded file.",
                "disclaimer": disclaimer,
            }

        gemini_result = await self._call_gemini(gemini_prompt)
        if not gemini_result or not gemini_result.get("ok") or not gemini_result.get("answer"):
            raise RuntimeError(f"Gemini failed: {(gemini_result or {}).get('error') or 'unknown_error'}")

        return {
            "answer": gemini_result.get("answer", ""),
            "confidence": float(gemini_result.get("confidence", 0.0)),
            "model_used": "Gemini",
            "actual_model_used": str(gemini_result.get("model") or "Gemini"),
            "selection_reason": "User selected Gemini, so no other model was used.",
            "disclaimer": disclaimer,
        }
