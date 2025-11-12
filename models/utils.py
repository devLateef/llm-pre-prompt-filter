import re
from faker import Faker

faker = Faker()

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{1,30}\b")
NUMBER_RE = re.compile(r"\b\d{4,}\b")

def mask_pii(text: str):
    """Return (masked_text, entities_list) where entities_list is list of (original, label)."""
    entities = []
    masked = text

    for m in EMAIL_RE.finditer(text):
        entities.append((m.group(0), "EMAIL"))
    masked = EMAIL_RE.sub("[MASKED_EMAIL]", masked)

    for m in PHONE_RE.finditer(text):
        entities.append((m.group(0), "PHONE"))
    masked = PHONE_RE.sub("[MASKED_PHONE]", masked)

    for m in SSN_RE.finditer(text):
        entities.append((m.group(0), "SSN"))
    masked = SSN_RE.sub("[MASKED_SSN]", masked)

    for m in IBAN_RE.finditer(text):
        entities.append((m.group(0), "IBAN"))
    masked = IBAN_RE.sub("[MASKED_IBAN]", masked)

    for m in NUMBER_RE.finditer(text):
        entities.append((m.group(0), "NUMBER"))
    masked = NUMBER_RE.sub("[MASKED_NUMBER]", masked)

    return masked, entities

def replace_entities_with_fakes(masked_text: str, entities):
    """
    entities: list of (original, label)
    Replace each original first occurrence in order with plausible fake using Faker.
    """
    out = masked_text
    for orig, label in entities:
        if label == "EMAIL":
            fake = faker.email()
        elif label == "PHONE":
            fake = faker.phone_number()
        elif label == "SSN":
            try:
                fake = faker.ssn()
            except Exception:
                fake = "000-00-0000"
        elif label == "IBAN":
            try:
                fake = faker.iban()
            except Exception:
                fake = "XX00" + str(faker.random_number(digits=8))
        elif label == "NUMBER":
            fake = str(faker.random_number(digits=8))
        else:
            fake = "[REDACTED]"
        out = out.replace(orig, fake, 1)
        out = out.replace("[MASKED_EMAIL]", fake, 1) if "[MASKED_EMAIL]" in out else out
        out = out.replace("[MASKED_PHONE]", fake, 1) if "[MASKED_PHONE]" in out else out
    return out