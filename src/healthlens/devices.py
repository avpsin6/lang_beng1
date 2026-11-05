"""Recognising where each sample came from.

Every Apple Health record carries a ``sourceName`` (e.g. "Akshaj’s Apple
Watch", "AutoSleep") and usually a ``device`` description such as::

    <<HKDevice: 0x2834>, name:Apple Watch, manufacturer:Apple Inc.,
      model:Watch, hardware:Watch4,2, software:7.0>

We reduce that to a *kind* (watch / phone / app / manual) which drives
de-duplication priority, and a friendly model name for the report.
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

_HARDWARE_RE = re.compile(r"hardware:([A-Za-z]+\d+,\d+)")
_MODEL_RE = re.compile(r"model:([^,>]+)")

WATCH_MODELS: Dict[str, str] = {
    "Watch1,1": "Apple Watch (1st gen) 38mm", "Watch1,2": "Apple Watch (1st gen) 42mm",
    "Watch2,6": "Apple Watch Series 1 38mm", "Watch2,7": "Apple Watch Series 1 42mm",
    "Watch2,3": "Apple Watch Series 2 38mm", "Watch2,4": "Apple Watch Series 2 42mm",
    "Watch3,1": "Apple Watch Series 3 38mm (cellular)", "Watch3,2": "Apple Watch Series 3 42mm (cellular)",
    "Watch3,3": "Apple Watch Series 3 38mm", "Watch3,4": "Apple Watch Series 3 42mm",
    "Watch4,1": "Apple Watch Series 4 40mm", "Watch4,2": "Apple Watch Series 4 44mm",
    "Watch4,3": "Apple Watch Series 4 40mm (cellular)", "Watch4,4": "Apple Watch Series 4 44mm (cellular)",
    "Watch5,1": "Apple Watch Series 5 40mm", "Watch5,2": "Apple Watch Series 5 44mm",
    "Watch5,3": "Apple Watch Series 5 40mm (cellular)", "Watch5,4": "Apple Watch Series 5 44mm (cellular)",
    "Watch5,9": "Apple Watch SE 40mm", "Watch5,10": "Apple Watch SE 44mm",
    "Watch5,11": "Apple Watch SE 40mm (cellular)", "Watch5,12": "Apple Watch SE 44mm (cellular)",
    "Watch6,1": "Apple Watch Series 6 40mm", "Watch6,2": "Apple Watch Series 6 44mm",
    "Watch6,3": "Apple Watch Series 6 40mm (cellular)", "Watch6,4": "Apple Watch Series 6 44mm (cellular)",
    "Watch6,6": "Apple Watch Series 7 41mm", "Watch6,7": "Apple Watch Series 7 45mm",
    "Watch6,8": "Apple Watch Series 7 41mm (cellular)", "Watch6,9": "Apple Watch Series 7 45mm (cellular)",
    "Watch6,10": "Apple Watch SE (2nd gen) 40mm", "Watch6,11": "Apple Watch SE (2nd gen) 44mm",
    "Watch6,12": "Apple Watch SE (2nd gen) 40mm (cellular)", "Watch6,13": "Apple Watch SE (2nd gen) 44mm (cellular)",
    "Watch6,14": "Apple Watch Series 8 41mm", "Watch6,15": "Apple Watch Series 8 45mm",
    "Watch6,16": "Apple Watch Series 8 41mm (cellular)", "Watch6,17": "Apple Watch Series 8 45mm (cellular)",
    "Watch6,18": "Apple Watch Ultra",
    "Watch7,1": "Apple Watch Series 9 41mm", "Watch7,2": "Apple Watch Series 9 45mm",
    "Watch7,3": "Apple Watch Series 9 41mm (cellular)", "Watch7,4": "Apple Watch Series 9 45mm (cellular)",
    "Watch7,5": "Apple Watch Ultra 2",
    "Watch7,8": "Apple Watch Series 10 42mm", "Watch7,9": "Apple Watch Series 10 46mm",
    "Watch7,10": "Apple Watch Series 10 42mm (cellular)", "Watch7,11": "Apple Watch Series 10 46mm (cellular)",
}

PHONE_MODELS: Dict[str, str] = {
    "iPhone7,2": "iPhone 6", "iPhone7,1": "iPhone 6 Plus",
    "iPhone8,1": "iPhone 6s", "iPhone8,2": "iPhone 6s Plus", "iPhone8,4": "iPhone SE (1st gen)",
    "iPhone9,1": "iPhone 7", "iPhone9,3": "iPhone 7", "iPhone9,2": "iPhone 7 Plus", "iPhone9,4": "iPhone 7 Plus",
    "iPhone10,1": "iPhone 8", "iPhone10,4": "iPhone 8", "iPhone10,2": "iPhone 8 Plus", "iPhone10,5": "iPhone 8 Plus",
    "iPhone10,3": "iPhone X", "iPhone10,6": "iPhone X",
    "iPhone11,2": "iPhone XS", "iPhone11,4": "iPhone XS Max", "iPhone11,6": "iPhone XS Max",
    "iPhone11,8": "iPhone XR",
    "iPhone12,1": "iPhone 11", "iPhone12,3": "iPhone 11 Pro", "iPhone12,5": "iPhone 11 Pro Max",
    "iPhone12,8": "iPhone SE (2nd gen)",
    "iPhone13,1": "iPhone 12 mini", "iPhone13,2": "iPhone 12", "iPhone13,3": "iPhone 12 Pro",
    "iPhone13,4": "iPhone 12 Pro Max",
    "iPhone14,4": "iPhone 13 mini", "iPhone14,5": "iPhone 13", "iPhone14,2": "iPhone 13 Pro",
    "iPhone14,3": "iPhone 13 Pro Max", "iPhone14,6": "iPhone SE (3rd gen)",
    "iPhone14,7": "iPhone 14", "iPhone14,8": "iPhone 14 Plus", "iPhone15,2": "iPhone 14 Pro",
    "iPhone15,3": "iPhone 14 Pro Max",
    "iPhone15,4": "iPhone 15", "iPhone15,5": "iPhone 15 Plus", "iPhone16,1": "iPhone 15 Pro",
    "iPhone16,2": "iPhone 15 Pro Max",
    "iPhone17,3": "iPhone 16", "iPhone17,4": "iPhone 16 Plus", "iPhone17,1": "iPhone 16 Pro",
    "iPhone17,2": "iPhone 16 Pro Max", "iPhone17,5": "iPhone 16e",
}

# Lower number = preferred when two sources measured the same period.
KIND_PRIORITY = {"watch": 1, "phone": 2, "app": 3, "manual": 4}


def device_tail(device: Optional[str]) -> str:
    """Strip the per-object memory address so equal devices compare equal."""
    if not device:
        return ""
    i = device.find(">")
    return device[i + 1:] if i >= 0 else device


def parse_device(device: Optional[str]) -> Tuple[str, str]:
    """Return (model, hardware) from an HKDevice description."""
    if not device:
        return "", ""
    hw = _HARDWARE_RE.search(device)
    model = _MODEL_RE.search(device)
    return (model.group(1).strip() if model else "", hw.group(1) if hw else "")


def classify(source_name: str, model: str, hardware: str) -> str:
    """watch | phone | app | manual."""
    name = source_name or ""
    if hardware.startswith("Watch") or model == "Watch":
        return "watch"
    if hardware.startswith("iPhone") or model == "iPhone":
        return "phone"
    if name == "Health":
        return "manual"
    if name == "Clock":  # iOS 13 and earlier Bedtime feature
        return "phone"
    low = name.lower()
    if "apple watch" in low or low.endswith(" watch") or "’s watch" in low or "'s watch" in low:
        return "watch"
    if "iphone" in low:
        return "phone"
    return "app"


def friendly_model(model: str, hardware: str) -> str:
    if hardware in WATCH_MODELS:
        return WATCH_MODELS[hardware]
    if hardware in PHONE_MODELS:
        return PHONE_MODELS[hardware]
    if hardware:
        return f"{model or 'Device'} ({hardware})"
    return model or ""


def watch_generation(hardware: str) -> Optional[int]:
    """Series number for capability hints (1st gen = 0, SE ~ 6, Ultra ~ 8)."""
    name = WATCH_MODELS.get(hardware, "")
    m = re.search(r"Series (\d+)", name)
    if m:
        return int(m.group(1))
    if "(1st gen)" in name:
        return 0
    if "SE (2nd gen)" in name:
        return 8
    if "SE" in name:
        return 6
    if "Ultra" in name:
        return 8 if name.endswith("Ultra") else 9
    return None
