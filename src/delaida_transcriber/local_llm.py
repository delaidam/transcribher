"""Small Ollama client used for optional, fully local transcript cleanup."""

import json
from urllib.error import URLError
from urllib.request import Request, urlopen


def refine_transcript(text: str, model: str, base_url: str) -> dict[str, object]:
    prompt = f"""Ti si pažljivi urednik bosanskog transkripta.

Uredi tekst ispod, ali strogo poštuj ova pravila:
- Ne dodaj činjenice, imena ili riječi koje nisu vjerovatno izgovorene.
- Ispravi samo očigledne greške prepoznavanja, pravopis i interpunkciju.
- Ako nisi gotovo potpuno siguran da je riječ pogrešno prepoznata, NE mijenjaj je.
- Ako neki dio nije jasan, zadrži originalnu riječ i dodaj [nejasno] umjesto nagađanja.
- Zadrži smisao, redoslijed i stil govornika.
- Napravi kratak TL;DR i listu ključnih tačaka samo na osnovu teksta.

Vrati isključivo JSON sa poljima:
cleaned_text (string), summary (string), key_points (lista stringova),
unclear_parts (lista stringova).

SIROVI TRANSKRIPT:
{text}
"""
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "seed": 0, "num_predict": 2048},
        }
    ).encode()
    request = Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=900) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (URLError, TimeoutError) as error:
        raise RuntimeError(
            "Ollama nije dostupna ili je obrada istekla. Provjeri da 'ollama serve' radi "
            "i da model postoji."
        ) from error
    try:
        result = json.loads(body["response"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Ollama je vratila neispravan odgovor.") from error
    if not isinstance(result, dict) or "cleaned_text" not in result:
        raise RuntimeError("Ollama odgovor nema očekivana polja.")
    return result
