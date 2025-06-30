import re
import langid
import os

input_filename = "Dogovor.txt"
if not os.path.exists(input_filename):
    raise FileNotFoundError(f"Файл {input_filename} не найден.")

with open(input_filename, "r", encoding="utf-8") as file:
    text = file.read()


paragraphs = re.split(r'\n\s*\n+', text)
paragraphs = [p.strip() for p in paragraphs if p.strip()]


def get_key_words(text):
    return [w.lower() for w in re.findall(r'\w+', text) if len(w) > 3]


langs = []
for p in paragraphs:
    try:
        lang, _ = langid.classify(p)
    except Exception:
        lang = "unknown"
    langs.append(lang)


for i, p in enumerate(paragraphs):
    if len(p) < 30:
        key_words = get_key_words(p)
        found_match = False
        for j, big_p in enumerate(paragraphs):
            if i != j and len(big_p) > len(p):
                if p in big_p:
                    langs[i] = langs[j]
                    found_match = True
                    break
                elif any(re.search(r'\b' + re.escape(kw) + r'\b', big_p, flags=re.IGNORECASE) for kw in key_words):
                    langs[i] = langs[j]
                    found_match = True
                    break

        if not found_match:
            extended_text = (p + " ") * 3 
            try:
                lang, _ = langid.classify(extended_text.strip())
                langs[i] = lang
            except Exception:
                pass

for lang in set(langs):
    output_filename = f"{lang}.txt"
    if os.path.exists(output_filename):
        os.remove(output_filename)


for lang, para in zip(langs, paragraphs):
    output_filename = f"{lang}.txt"
    with open(output_filename, "a", encoding="utf-8") as out_file:
        out_file.write(para + "\n\n")
