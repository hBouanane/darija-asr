from darija_asr.normalize import latin_ratio, normalize


def test_hamza_and_alef_forms_unified():
    assert normalize("أنا إلى آخر") == normalize("انا الي اخر")


def test_ta_marbuta_alef_maqsura_tashkeel_tatweel():
    assert normalize("مَزْيَانــة") == "مزيانه"
    assert normalize("على") == "علي"


def test_punctuation_digits_and_latin_case():
    assert normalize("واخا، Merci! ٣") == "واخا merci 3"
    assert normalize("  bonjour   ça va؟ ") == "bonjour ca va"


def test_empty_inputs():
    assert normalize(None) == ""
    assert normalize("") == ""


def test_latin_ratio():
    assert latin_ratio("merci") == 1.0
    assert latin_ratio("شكرا") == 0.0
    assert 0 < latin_ratio("واخا merci") < 1
