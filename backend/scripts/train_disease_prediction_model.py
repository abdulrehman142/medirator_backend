"""
Training script for disease prediction models.

This script generates the required ML models for disease prediction:
- symptom_model.pkl: RandomForestClassifier
- symptoms_list.pkl: List of 377 symptoms
- label_encoder.pkl: LabelEncoder for disease names
"""

import os
import pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json

# Define 377 symptoms
SYMPTOMS = [
    "fever", "cough", "shortness of breath", "chest pain", "headache",
    "fatigue", "muscle ache", "loss of appetite", "sore throat", "nausea",
    "vomiting", "diarrhea", "abdominal pain", "constipation", "bloating",
    "gas", "heartburn", "acid reflux", "indigestion", "dizziness",
    "vertigo", "blurred vision", "double vision", "eye pain", "dry eyes",
    "tearing", "itchy eyes", "ear pain", "ear fullness", "tinnitus",
    "hearing loss", "nasal congestion", "runny nose", "sneezing", "nose bleed",
    "sinus pain", "sinus drainage", "bad breath", "mouth ulcers", "gum swelling",
    "gum bleeding", "tooth pain", "tooth sensitivity", "loose teeth", "jaw pain",
    "difficulty chewing", "difficulty swallowing", "hoarseness", "voice loss", "whooping cough",
    "asthma", "wheezing", "chest tightness", "shallow breathing", "rapid breathing",
    "slow breathing", "breath holding", "hiccups", "sighing", "yawning",
    "sleep apnea", "snoring", "sleepiness", "insomnia", "nightmares",
    "night sweats", "sleepwalking", "sleep paralysis", "sleep talking", "tremor",
    "shaking", "stiffness", "muscle cramps", "muscle weakness", "muscle twitching",
    "numbness", "tingling", "burning sensation", "electric shock sensation", "pain",
    "aching", "soreness", "tenderness", "bruising", "swelling",
    "inflammation", "redness", "warmth", "coldness", "pallor",
    "flushing", "cyanosis", "jaundice", "hives", "rash",
    "itching", "eczema", "psoriasis", "acne", "boils",
    "carbuncles", "skin ulcers", "warts", "moles", "skin tags",
    "freckles", "age spots", "vitiligo", "alopecia", "hair loss",
    "excessive hair", "hirsutism", "dandruff", "scalp itching", "scalp pain",
    "ingrown nails", "nail discoloration", "nail brittleness", "nail ridges", "clubbing",
    "edema", "lymph node swelling", "spleen enlargement", "liver enlargement", "thyroid enlargement",
    "goiter", "breast tenderness", "breast swelling", "nipple discharge", "nipple inversion",
    "testicular pain", "testicular swelling", "prostate pain", "pelvic pain", "lower back pain",
    "upper back pain", "mid back pain", "neck pain", "shoulder pain", "arm pain",
    "elbow pain", "wrist pain", "hand pain", "finger pain", "hip pain",
    "thigh pain", "knee pain", "shin pain", "ankle pain", "foot pain",
    "heel pain", "toe pain", "joint pain", "joint stiffness", "reduced range of motion",
    "clicking joints", "popping joints", "locking joints", "unstable joints", "overactive reflexes",
    "underactive reflexes", "abnormal reflexes", "seizures", "convulsions", "fainting",
    "syncope", "blackouts", "loss of consciousness", "confusion", "delirium",
    "dementia", "memory loss", "difficulty concentrating", "brain fog", "forgetfulness",
    "poor judgment", "impaired decision making", "anxiety", "panic attacks", "phobias",
    "compulsive behavior", "obsessive thoughts", "paranoia", "hallucinations", "delusions",
    "mood swings", "irritability", "anger", "aggression", "sadness",
    "depression", "hopelessness", "suicidal thoughts", "low self esteem", "shame",
    "guilt", "apathy", "anhedonia", "withdrawal", "isolation",
    "hyperactivity", "impulsivity", "inattention", "distractibility", "restlessness",
    "fidgeting", "pacing", "repetitive behavior", "tics", "stuttering",
    "slurred speech", "mutism", "selective mutism", "articulation disorder", "voice changes",
    "high pitched voice", "low pitched voice", "nasal voice", "breathy voice", "raspy voice",
    "monotone voice", "emotional lability", "inappropriate laughter", "crying", "sudden mood changes",
    "exaggerated reactions", "diminished reactions", "flat affect", "blunted affect", "restricted affect",
    "inappropriate affect", "incongruent affect", "dissociation", "derealization", "depersonalization",
    "out of body experience", "time distortion", "lost time", "gaps in memory", "recovered memories",
    "confabulation", "false memories", "intrusive memories", "flashbacks", "nightmares",
    "sleep deprivation", "fatigue", "exhaustion", "weakness", "lethargy",
    "malaise", "unwell feeling", "aching all over", "heavy feeling", "light feeling",
    "floating sensation", "sinking sensation", "crushing sensation", "pressure", "tightness",
    "tension", "rigidity", "paralysis", "temporary paralysis", "hemiplegia",
    "paraplegia", "tetraplegia", "quadriplegia", "monoplegia", "drooping",
    "sagging", "asymmetry", "deviation", "deviation of mouth", "deviation of eyes",
    "ptosis", "lid lag", "nystagmus", "strabismus", "diplopia",
    "scotoma", "floaters", "flashing lights", "halos", "metamorphopsia",
    "micropsia", "macropsia", "akinetopsia", "color blindness", "night blindness",
    "photophobia", "photopsia", "visual snow", "tunnel vision", "peripheral vision loss",
    "central vision loss", "monocular vision loss", "binocular vision loss", "sudden blindness", "gradual blindness",
    "ocular pain", "foreign body sensation", "sand in eyes feeling", "gritty sensation", "dry sensation",
    "watery eyes", "excessive tearing", "red eyes", "yellow eyes", "white eyes",
    "bloodshot eyes", "subconjunctival hemorrhage", "hyphema", "corneal abrasion", "corneal ulcer",
    "cataracts", "glaucoma", "retinal detachment", "macular degeneration", "retinitis",
    "uveitis", "keratitis", "conjunctivitis", "blepharitis", "chalazion",
    "stye", "pinguecula", "pterygium", "presbyopia", "myopia",
    "hyperopia", "astigmatism", "anisometropia", "amblyopia", "lazy eye",
    "color vision deficiency", "protanopia", "deuteranopia", "tritanopia", "achromatopsia",
    "diplopia", "monocular diplopia", "binocular diplopia", "oscillopsia", "vertigo",
    "lightheadedness", "presyncope", "postural hypotension", "orthostatic intolerance", "syncope",
    "tachycardia", "bradycardia", "palpitations", "arrhythmia", "atrial fibrillation",
    "flutter", "premature beats", "skipped beats", "irregular heartbeat", "irregular pulse",
    "weak pulse", "bounding pulse", "collapsing pulse", "water hammer pulse", "bigeminy",
    "trigeminy", "couplets", "runs", "salvo", "burst",
    "episode", "paroxysm", "attack", "spell", "fit",
    "turn", "dizzy spell", "fainting spell", "anxiety attack", "panic episode"
]

# Define diseases for classification
DISEASES = [
    "Common Cold",
    "Influenza",
    "COVID-19",
    "Pneumonia",
    "Bronchitis",
    "Asthma",
    "COPD",
    "Allergic Rhinitis",
    "Sinusitis",
    "Laryngitis",
    "Pharyngitis",
    "Tonsillitis",
    "Otitis Media",
    "Otitis Externa",
    "Otosclerosis",
    "Tinnitus Disorder",
    "Hearing Loss",
    "Vertigo",
    "Meniere's Disease",
    "BPPV",
    "Conjunctivitis",
    "Keratitis",
    "Uveitis",
    "Glaucoma",
    "Cataracts",
    "Macular Degeneration",
    "Retinal Detachment",
    "Color Blindness",
    "Refractive Error",
    "Presbyopia",
    "Myopia",
    "Hyperopia",
    "Astigmatism",
    "Migraine",
    "Tension Headache",
    "Cluster Headache",
    "Trigeminal Neuralgia",
    "Bell's Palsy",
    "Parkinson's Disease",
    "Alzheimer's Disease",
    "Dementia",
    "Multiple Sclerosis",
    "ALS",
    "Epilepsy",
    "Seizure Disorder",
    "Stroke",
    "TIA",
    "Aneurysm",
    "Brain Tumor",
    "Meningitis",
    "Encephalitis",
    "Depression",
    "Anxiety Disorder",
    "Bipolar Disorder",
    "Schizophrenia",
    "OCD",
    "PTSD",
    "ADHD",
    "Autism Spectrum",
    "PANS",
    "PANDAS",
    "Personality Disorder",
    "Heart Disease",
    "Hypertension",
    "Hypotension",
    "Arrhythmia",
    "Atrial Fibrillation",
    "Heart Failure",
    "Cardiomyopathy",
    "Myocarditis",
    "Pericarditis",
    "Angina",
    "Myocardial Infarction",
    "Coronary Artery Disease",
    "Atherosclerosis",
    "Arteriosclerosis",
    "Aneurysm",
    "Peripheral Artery Disease",
    "Venous Thromboembolism",
    "Deep Vein Thrombosis",
    "Pulmonary Embolism",
    "Varicose Veins",
    "Hemorrhoids",
    "Diabetes Type 1",
    "Diabetes Type 2",
    "Gestational Diabetes",
    "Prediabetes",
    "Hypoglycemia",
    "Hyperglycemia",
    "Diabetic Neuropathy",
    "Diabetic Retinopathy",
    "Diabetic Nephropathy",
    "Diabetic Ketoacidosis",
    "Obesity",
    "Overweight",
    "Metabolic Syndrome",
    "Thyroid Disease",
    "Hypothyroidism",
    "Hyperthyroidism",
    "Thyroiditis",
    "Thyroid Cancer",
    "Goiter",
    "Graves' Disease",
    "Hashimoto's Thyroiditis",
    "Adrenal Insufficiency",
    "Cushing's Syndrome",
    "PCOS",
    "Menopause",
    "Hormone Imbalance",
    "Gastroesophageal Reflux",
    "Peptic Ulcer Disease",
    "Gastritis",
    "Gastroparesis",
    "Celiac Disease",
    "Crohn's Disease",
    "Ulcerative Colitis",
    "Irritable Bowel Syndrome",
    "Inflammatory Bowel Disease",
    "Diverticulitis",
    "Appendicitis",
    "Cholecystitis",
    "Cholelithiasis",
    "Pancreatitis",
    "Fatty Liver Disease",
    "Hepatitis",
    "Cirrhosis",
    "Liver Cancer",
    "Hepatic Encephalopathy",
    "Portal Hypertension",
    "Kidney Disease",
    "Kidney Failure",
    "Nephrotic Syndrome",
    "Nephritic Syndrome",
    "Glomerulonephritis",
    "Chronic Kidney Disease",
    "Acute Kidney Injury",
    "Kidney Stones",
    "Urinary Tract Infection",
    "Cystitis",
    "Pyelonephritis",
    "Prostate Cancer",
    "Benign Prostatic Hyperplasia",
    "Prostatitis",
    "Erectile Dysfunction",
    "Premature Ejaculation",
    "Low Testosterone",
    "Infertility",
    "Testicular Cancer",
    "Ovarian Cancer",
    "Uterine Cancer",
    "Cervical Cancer",
    "Breast Cancer",
    "Endometriosis",
    "Uterine Fibroids",
    "Polycystic Ovary Syndrome",
    "Menopausal Symptoms",
    "Vaginal Atrophy",
    "Osteoporosis",
    "Osteoarthritis",
    "Rheumatoid Arthritis",
    "Gout",
    "Lupus",
    "Scleroderma",
    "Sjögren's Syndrome",
    "Vasculitis",
    "Behçet's Disease",
    "Ankylosing Spondylitis",
    "Psoriatic Arthritis",
    "Reactive Arthritis",
    "Fibromyalgia",
    "Chronic Fatigue Syndrome",
    "Myalgic Encephalomyelitis",
    "Lyme Disease",
    "Chronic Lyme Disease",
    "Rheumatic Fever",
    "Joint Hypermobility",
    "Ehlers-Danlos Syndrome",
    "Marfan Syndrome",
    "Osteogenesis Imperfecta",
    "Sickle Cell Disease",
    "Thalassemia",
    "Hemophilia",
    "Von Willebrand Disease",
    "Thrombophilia",
    "Platelet Disorder",
    "Leukemia",
    "Lymphoma",
    "Myeloma",
    "Anemia",
    "Iron Deficiency Anemia",
    "Vitamin B12 Deficiency",
    "Folate Deficiency",
    "Pernicious Anemia",
    "Aplastic Anemia",
    "Hemolytic Anemia",
    "Thalassemia Major",
    "Thalassemia Minor",
    "G6PD Deficiency",
    "Immunodeficiency",
    "HIV/AIDS",
    "Acute Immune Deficiency",
    "Severe Combined Immunodeficiency",
    "Selective IgA Deficiency",
    "Agammaglobulinemia",
    "Cancer",
    "Melanoma",
    "Basal Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Lung Cancer",
    "Pancreatic Cancer",
    "Colon Cancer",
    "Rectal Cancer",
    "Gastric Cancer",
    "Esophageal Cancer",
    "Mouth Cancer",
    "Throat Cancer",
    "Laryngeal Cancer",
    "Thyroid Cancer",
    "Endocrine Cancer",
    "Skin Cancer",
    "Mesothelioma",
    "Unknown Cancer",
    "Benign Tumor",
    "Cyst",
    "Polyp",
    "Lipoma",
    "Wart",
    "Mole",
    "Nevus",
    "Keloid",
    "Scar",
    "Burn",
    "Frostbite",
    "Sunburn",
    "Chemical Burn",
    "Acne",
    "Rosacea",
    "Seborrhea",
    "Psoriasis",
    "Lichen Planus",
    "Pityriasis Rosea",
    "Vitiligo",
    "Albinism",
    "Melasma",
    "Eczema",
    "Dermatitis",
    "Contact Dermatitis",
    "Atopic Dermatitis",
    "Seborrheic Dermatitis",
    "Dyshidrotic Dermatitis",
    "Nummular Dermatitis",
    "Photodermatitis",
    "Urticaria",
    "Angioedema",
    "Pruritus",
    "Prurigo",
    "Scabies",
    "Lice",
    "Ringworm",
    "Athlete's Foot",
    "Jock Itch",
    "Yeast Infection",
    "Candidiasis",
    "Pityriasis Versicolor",
    "Onychomycosis",
    "Impetigo",
    "Cellulitis",
    "Abscess",
    "Boil",
    "Carbuncle",
    "Folliculitis",
    "Hidradenitis",
    "Infected Hair Follicle",
    "Burrow",
    "Erythrasma",
    "Pseudomonas Infection",
    "Herpes Simplex",
    "Herpes Zoster",
    "Shingles",
    "Varicella",
    "Chickenpox",
    "Molluscum Contagiosum",
    "Wart",
    "Common Wart",
    "Plantar Wart",
    "Genital Wart",
    "Flat Wart",
    "Filiform Wart",
    "Periungual Wart",
    "Verruca Vulgaris",
    "Verruca Plantaris",
    "Seborrheic Keratosis",
    "Acanthosis Nigricans",
    "Skin Tag",
    "Acrochordons",
    "Hemangioma",
    "Port Wine Stain",
    "Spider Angioma",
    "Cherry Angioma",
    "Telangiectasia",
    "Capillary Malformation",
    "Lymphangioma",
    "Cystic Hygroma",
    "Hair Loss",
    "Male Pattern Baldness",
    "Female Pattern Baldness",
    "Alopecia Areata",
    "Telogen Effluvium",
    "Anagen Effluvium",
    "Trichotillomania",
    "Minoxidil Syndrome",
    "Finasteride Syndrome"
]

def create_training_data(n_samples=500):
    """Create dummy training data for disease prediction."""
    np.random.seed(42)
    
    X = np.random.randint(0, 2, size=(n_samples, len(SYMPTOMS)))
    y = np.random.choice(DISEASES, size=n_samples)
    
    return X, y

def train_models(output_dir="models"):
    """Train and save the disease prediction models."""
    print("🏥 Disease Prediction Model Training")
    print("=" * 50)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Generate training data
    print(f"\n📊 Generating {len(SYMPTOMS)} symptoms and {len(DISEASES)} diseases...")
    
    # Train label encoder first
    print("\n🏷️  Training LabelEncoder...")
    label_encoder = LabelEncoder()
    label_encoder.fit(DISEASES)
    print(f"✓ Label encoder trained for {len(label_encoder.classes_)} diseases")
    
    # Generate training data using encoded labels
    np.random.seed(42)
    n_samples = 500
    X = np.random.randint(0, 2, size=(n_samples, len(SYMPTOMS)))
    y = np.random.choice(label_encoder.classes_, size=n_samples)
    print(f"✓ Generated {len(X)} training samples")
    
    # Train model
    print("\n🤖 Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    print("✓ Model trained")
    
    # Save models
    print("\n💾 Saving models...")
    
    model_path = Path(output_dir) / "symptom_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Saved symptom_model.pkl ({model_path.stat().st_size / 1024:.1f} KB)")
    
    symptoms_path = Path(output_dir) / "symptoms_list.pkl"
    with open(symptoms_path, "wb") as f:
        pickle.dump(SYMPTOMS, f)
    print(f"✓ Saved symptoms_list.pkl ({symptoms_path.stat().st_size / 1024:.1f} KB)")
    
    encoder_path = Path(output_dir) / "label_encoder.pkl"
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"✓ Saved label_encoder.pkl ({encoder_path.stat().st_size / 1024:.1f} KB)")
    
    # Save metadata
    metadata = {
        "n_symptoms": len(SYMPTOMS),
        "n_diseases": len(DISEASES),
        "n_samples": len(X),
        "diseases": DISEASES,
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata.json")
    
    print("\n" + "=" * 50)
    print("✅ Models trained and saved successfully!")
    print(f"📁 Output directory: {Path(output_dir).absolute()}")
    print("\n📋 Summary:")
    print(f"   • Symptoms: {len(SYMPTOMS)}")
    print(f"   • Diseases: {len(DISEASES)}")
    print(f"   • Training samples: {len(X)}")
    print(f"   • Model type: RandomForestClassifier")
    print(f"   • Random state: 42")
    
    return model, SYMPTOMS, label_encoder

if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "models"
    train_models(output_dir)
