# Darija transcription style guide (DRAFT — edit to match your conventions)

This guide is sent to the LLM judge with every request. Every rule here becomes
a convention of the final dataset, so change it before the large runs, not after.

## Faithfulness
- Write exactly what is spoken. Never translate Darija into MSA, never "correct"
  grammar, never complete cut-off words or sentences.
- Keep repetitions and false starts that are real words ("كنت كنت غادي").
- Drop pure non-lexical hesitation sounds (ممم، euh, aah).

## Script
- Darija and Arabic words: Arabic script.
- French, English and Spanish words: Latin script, standard spelling of that
  language, lowercase except proper nouns ("merci", "normal", "le weekend",
  "meeting").
- Words borrowed long ago and pronounced the Darija way stay in Arabic script
  (طوموبيل، فريج، بلاصة).
- Brand and place names: the script and spelling most commonly used for them.

## Spelling
- Use ڭ for the /g/ sound (ڭال، ڭاع).
- Feminine ending: ة (مزيانة، الطوموبيلة).
- Present-tense prefix: كا / ك as spoken, attached to the verb (كنمشي، كاتعرف).
- Future marker: غادي / غا, written as spoken.
- Negation: ما...ش attached (ماعرفتش، ماكاينش).
- No diacritics (tashkeel), no tatweel.

## Gender and number endings
Speech recognizers often drop short word endings, so a candidate may show the bare
masculine form where the speaker said the feminine or the plural.
- Always write the ending that marks gender or number on adjectives, participles and
  verbs: feminine ة (pronounced -a), plural ين (Darija uses it for both genders).
  هارب (he) / هاربة (she, a country) / هاربين (they, countries); جاي / جاية / جايين;
  مزيان / مزيانة / مزيانين.
- When candidates differ only in such an ending, choose the one that agrees with the
  word it describes: الدولة هاربة، الدول هاربين، البنت جاية، الدراري جايين.
- When no candidate has the ending but the agreement is clear from the sentence (the
  noun or pronoun it refers to is in the same clip), add it: this restores a sound the
  recognizers dropped, it is not new content. When the agreement is not clear, keep
  the form the candidates support.

## Numbers
- Write numbers as words, as spoken ("جوج", "ميا وعشرين", "deux mille").

## Punctuation
- No punctuation.
