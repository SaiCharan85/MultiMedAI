"""Shared analysis layer — model loading + core capabilities, UI-agnostic.

Extracted from the old Streamlit app so the FastAPI backend (and anything else)
can reuse the exact same logic: retrieval, modality detection, finding
classification, neighbour grounding, and image generation.
"""
from __future__ import annotations

import functools

from src.common import load_config, resolve, get_device

CFG = load_config()

SOURCE_INFO = {
    "radiology": ("ROCOv2 Radiology (open)",
                  "https://huggingface.co/datasets/eltorio/ROCOv2-radiology"),
    "pathology": ("PathVQA (open)",
                  "https://huggingface.co/datasets/flaviagiammarino/path-vqa"),
    "tb": ("TB/COVID/Pneumonia CXR (open)",
           "https://huggingface.co/datasets/DevVoyageR007/classify_Pneumonia_Tuberculosis_and_Normal__Non_Xray_chest_Xray_images"),
    "derm": ("Skin Cancer / Dermatology (open)",
             "https://huggingface.co/datasets/marmal88/skin_cancer"),
    "brainmri": ("Brain MRI / Alzheimer (open)",
                 "https://huggingface.co/datasets/Falah/Alzheimer_MRI"),
    "bone": ("Bone / Fracture X-ray (open)",
             "https://huggingface.co/datasets/Hemg/bone-fracture-detection"),
    "hair": ("Hair & Scalp Dermatology (open)",
             "https://huggingface.co/datasets/AbishekFranklin/medai-vision-dataset-hair_scalp_conditions"),
}


def provenance(meta):
    return SOURCE_INFO.get(meta.get("source", "pathology"), SOURCE_INFO["pathology"])


@functools.lru_cache(maxsize=1)
def _clip():
    from src.biomedclip import load_biomedclip
    return load_biomedclip(CFG["retrieval"]["clip_model"])


@functools.lru_cache(maxsize=1)
def _bank():
    from src.retrieval import load_bank
    return load_bank(CFG)


def retrieve(query, topk):
    from src.biomedclip import encode_texts
    model, preprocess, tokenizer, device = _clip()
    bank, metas = _bank()
    q = encode_texts(model, tokenizer, device, [query])
    sims = (q @ bank.T).squeeze(0)
    vals, idx = sims.topk(min(topk, len(metas)))
    bank_dir = resolve(CFG["paths"]["data"], "image_bank")
    return [(bank_dir / metas[i]["file"], metas[i], float(vals[k]))
            for k, i in enumerate(idx.tolist())]


def neighbors(pil_img, k=4):
    from src.biomedclip import encode_images
    model, preprocess, tokenizer, device = _clip()
    bank, metas = _bank()
    ie = encode_images(model, preprocess, device, [pil_img])
    sims = (ie @ bank.T).squeeze(0)
    vals, idx = sims.topk(min(k, len(metas)))
    return [(metas[i].get("question", ""), float(vals[j]))
            for j, i in enumerate(idx.tolist())]


MODALITIES = [
    "a brain MRI or CT scan", "a chest X-ray radiograph",
    "an abdominal or pelvic CT scan",
    "a histopathology microscope slide (H&E stained tissue)",
    "a retinal fundus photograph", "a bone or limb X-ray", "an ultrasound image",
]

FINDINGS = {
    "brain": ["a brain tumor or mass", "an intracranial hemorrhage",
              "an ischemic stroke / infarct", "white matter lesions",
              "hydrocephalus (enlarged ventricles)", "a normal brain"],
    "chest": ["pulmonary tuberculosis", "pneumonia", "COVID-19 pneumonia",
              "a lung mass or nodule", "pleural effusion", "cardiomegaly",
              "a normal chest"],
    "abdom": ["a mass or tumor", "bowel obstruction", "free fluid / ascites",
              "an abscess or collection", "normal abdomen"],
    "patho": ["carcinoma / malignant tumor tissue", "inflammation", "necrosis",
              "benign tissue", "normal tissue"],
    "bone": ["a fracture", "arthritis / joint degeneration", "a bone lesion",
             "normal bone"],
    "retina": ["diabetic retinopathy", "glaucoma", "macular degeneration",
               "a normal retina"],
    "ultra": ["a cyst", "a mass", "a stone / calculus", "normal ultrasound"],
}


def modality_key(modality: str) -> str:
    m = modality.lower()
    for kw, key in (("brain", "brain"), ("chest", "chest"), ("abdom", "abdom"),
                    ("pelvic", "abdom"), ("histopath", "patho"), ("slide", "patho"),
                    ("bone", "bone"), ("limb", "bone"), ("retina", "retina"),
                    ("fundus", "retina"), ("ultrasound", "ultra")):
        if kw in m:
            return key
    return "patho"


def modality_check(pil_img):
    from src.biomedclip import encode_images, encode_texts
    model, preprocess, tokenizer, device = _clip()
    ie = encode_images(model, preprocess, device, [pil_img])
    te = encode_texts(model, tokenizer, device, MODALITIES)
    probs = (ie @ te.T)[0].softmax(0)
    k = int(probs.argmax())
    return MODALITIES[k], float(probs[k])


def finding_check(pil_img, modality, topn=3):
    """(preds, reliable). Trained head if available (e.g. chest ~94%), else zero-shot."""
    from src.biomedclip import encode_images, encode_texts
    from src import findings
    model, preprocess, tokenizer, device = _clip()
    ie = encode_images(model, preprocess, device, [pil_img])
    key = modality_key(modality)
    if findings.available(key):
        return findings.predict(key, ie, topn=topn), True
    labels = FINDINGS[key]
    te = encode_texts(model, tokenizer, device,
                      [f"this medical image shows {l}" for l in labels])
    probs = (ie @ te.T)[0].softmax(0)
    order = probs.argsort(descending=True)[:topn].tolist()
    return [(labels[i], float(probs[i])) for i in order], False


def generate(prompt):
    import torch
    from diffusers import AutoPipelineForText2Image
    from src import engine
    scfg = CFG["synthesis"]
    weights = resolve(CFG["paths"]["weights"], scfg.get("weights_subdir", "sdturbo"))
    pipe = AutoPipelineForText2Image.from_pretrained(
        scfg["model_id"], torch_dtype=torch.float32, safety_checker=None,
        cache_dir=str(weights)).to(get_device())
    lp = engine.lora_path(CFG)
    if lp:
        pipe.load_lora_weights(str(lp.parent))
    full = f"{prompt}, {scfg.get('style_suffix', '')}".strip(", ")
    kw = dict(num_inference_steps=scfg["num_inference_steps"],
              guidance_scale=scfg["guidance_scale"],
              height=scfg["image_size"], width=scfg["image_size"])
    if scfg.get("guidance_scale", 0) > 1 and scfg.get("negative_prompt"):
        kw["negative_prompt"] = scfg["negative_prompt"]
    return pipe(full, **kw).images[0]
