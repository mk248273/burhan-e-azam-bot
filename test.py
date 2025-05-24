import langid

def is_input_in_target_language(user_input, target_language):
    lang_map = {
        "English": "en",
        "Arabic": "ar",
    }
    target_lang_code = lang_map.get(target_language)
    detected_lang, _ = langid.classify(user_input)
    return detected_lang == target_lang_code

# Test with single words
print(is_input_in_target_language("hello", "English"))       # True
print(is_input_in_target_language("سلام", "Arabic"))         # True
print(is_input_in_target_language("السلام", "Arabic"))       # True
print(is_input_in_target_language("محبت", "Arabic"))           # True (means love)
print(is_input_in_target_language("دوستی", "Arabic"))          # True (means friendship)
print(is_input_in_target_language("book", "Arabic"))           # False (English word)
print(is_input_in_target_language("کتاب", "Arabic"))        # False (Urdu word)
