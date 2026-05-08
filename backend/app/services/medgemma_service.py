import os
import platform
import torch
from huggingface_hub import login
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from app.core.config import get_settings
from app.vision import load_rgb_image


class MedGemmaMultimodalService:
    """
    MedGemma / Gemma3 multimodal via AutoProcessor + AutoModelForImageTextToText.
    This service loads the base MedGemma checkpoint only.
    """

    def __init__(self):
        settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.model_path = settings.medgemma_model_path
        self.hf_token = settings.huggingface_hub_token or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
        self.processor = None
        self.model = None
        self.mock_mode = False
        self.runtime_mode = "unknown"
        self.supports_vision = True
        self.load_error: str | None = None

    def _authenticate_hf(self) -> None:
        if self.hf_token:
            login(token=self.hf_token, add_to_git_credential=False)

    def _resolve_model_source(self):
        if os.path.isdir(self.model_path):
            return self.model_path, True
        local_candidate = os.path.join(self.base_dir, self.model_path)
        if os.path.isdir(local_candidate):
            return local_candidate, True
        return self.model_path, False

    def _calculate_confidence(self, text_answer: str, raw_model_prob: float = 0.85) -> float:
        heuristic_score = 0.0
        lower_ans = text_answer.lower()
        if "i'm not sure" in lower_ans or "maybe" in lower_ans or "i am not sure" in lower_ans:
            heuristic_score -= 0.3
        elif len(text_answer) < 30 or "unclear" in lower_ans:
            heuristic_score -= 0.2
        else:
            heuristic_score += 0.2

        confidence = (0.7 * raw_model_prob) + (0.3 * heuristic_score)
        return max(0.0, min(1.0, confidence))

    def _load_processor_hub_or_local_adapter(self, adapter_dir: str | None, base_repo: str) -> None:
        if adapter_dir is not None:
            try:
                self.processor = AutoProcessor.from_pretrained(
                    adapter_dir,
                    local_files_only=True,
                    token=self.hf_token,
                    trust_remote_code=True,
                )
                return
            except Exception:
                pass
        self.processor = AutoProcessor.from_pretrained(
            base_repo,
            local_files_only=False,
            token=self.hf_token,
            trust_remote_code=True,
        )

    def _load_base_multimodal(
        self,
        model_id: str,
        *,
        dtype_cuda,
        common_kw,
        is_macos: bool,
        has_cuda: bool,
        has_mps: bool,
    ):
        """Return ImageTextToText model (possibly quantized on CUDA)."""
        if has_cuda:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype_cuda,
                bnb_4bit_quant_type="nf4",
            )
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=dtype_cuda,
                **common_kw,
            )
            return model, "cuda_4bit_device_map_auto"

        if has_mps and is_macos:
            try:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    **common_kw,
                ).to("mps")
                return model, "mps_bf16_manual"
            except Exception:
                model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    **common_kw,
                ).to("cpu")
                return model, "cpu_float32_fallback"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            **common_kw,
        ).to("cpu")
        return model, "cpu_float32_manual"

    def load_model(self) -> None:
        self.load_error = None

        try:
            model_source, local_only = self._resolve_model_source()
            hub_auth_needed = not local_only

            if hub_auth_needed and not self.hf_token:
                raise RuntimeError(
                    "HF_TOKEN / HUGGINGFACE_HUB_TOKEN is required to download the multimodal "
                    "base checkpoint or to use gated repos."
                )
            if hub_auth_needed:
                self._authenticate_hf()

            dtype_cuda = torch.float16
            is_macos = platform.system().lower() == "darwin"
            has_cuda = torch.cuda.is_available()
            has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

            common_kw = dict(
                local_files_only=False,
                token=self.hf_token,
                trust_remote_code=True,
            )

            self.supports_vision = True

            ckpt_kw = dict(
                local_files_only=local_only,
                token=self.hf_token,
                trust_remote_code=True,
            )

            self.processor = AutoProcessor.from_pretrained(
                model_source,
                **ckpt_kw,
            )

            self.model, suffix = self._load_base_multimodal(
                model_source,
                dtype_cuda=dtype_cuda,
                common_kw=ckpt_kw,
                is_macos=is_macos,
                has_cuda=has_cuda,
                has_mps=has_mps,
            )
            self.runtime_mode = suffix
            self.mock_mode = False

        except Exception as e:
            self.supports_vision = True
            self.mock_mode = True
            self.runtime_mode = f"mock_mode:{e}"
            self.load_error = str(e)
            self.processor = None
            self.model = None

    def _build_prompt_inputs(self, text: str, image: Image.Image | None):
        messages: list[dict] = []

        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are MedGemma, a careful medical reasoning assistant. "
                            "Prefer facts from uploaded evidence when present; avoid hallucination; "
                            "state uncertainty and recommend clinician follow-up when appropriate."
                        ),
                    }
                ],
            }
        )

        if image is not None:
            user_content = [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ]
        else:
            user_content = [{"type": "text", "text": text}]

        messages.append({"role": "user", "content": user_content})

        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )

    def generate(self, text: str, image_path: str | None = None):
        if self.processor is None or self.model is None:
            self.load_model()

        if self.mock_mode or self.processor is None or self.model is None:
            import time

            time.sleep(0.3)
            conf = 0.85
            if image_path:
                answer_text = (
                    f"Mock Response: Simulated multimodal answer for: '{text[:200]}...' (image present)."
                )
            else:
                answer_text = f"Mock Response: Simulated answer for: '{text[:200]}...'"

            return {
                "answer": answer_text,
                "confidence": self._calculate_confidence(answer_text, raw_model_prob=conf),
                "reasoning": "Mock mode: model not loaded.",
            }

        if image_path and not self.supports_vision:
            text = (
                f"{text}\n[A vision input was supplied but this adapter is text-only adapter mode; describe limits.]"
            )
            image_path = None

        image: Image.Image | None = None
        if image_path:
            image = load_rgb_image(image_path)

        inputs = self._build_prompt_inputs(text, image)
        device = next(self.model.parameters()).device
        inputs = inputs.to(device)

        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None and "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model_dtype)

        input_len = int(inputs["input_ids"].shape[-1])

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        new_tokens = output_ids[0][input_len:]
        answer_text = self.processor.decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        confidence = self._calculate_confidence(answer_text, raw_model_prob=0.85)

        reasoning = (
            "Gemma3 multimodal generate (AutoModelForImageTextToText)."
        )

        return {
            "answer": answer_text,
            "confidence": confidence,
            "reasoning": reasoning,
        }


medgemma_service_instance = MedGemmaMultimodalService()
