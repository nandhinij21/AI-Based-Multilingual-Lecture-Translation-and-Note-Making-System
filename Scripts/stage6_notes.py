# ============================================================
# STAGE 6 : AI NOTE GENERATION SYSTEM
# Groq + Llama
# Complete multilingual educational notes pipeline
# ============================================================


import os
import json
import time
import re
from pathlib import Path

from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================


GROQ_API_KEY = "[REDACTED_GROQ]"

if not GROQ_API_KEY:
    raise ValueError("Please set GROQ_API_KEY environment variable")

MODEL_NAME = "llama-3.1-8b-instant"


OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


# Keep this small because Groq free tier has TPM limits

MAX_CHARS_PER_CHUNK = 1800


REQUEST_DELAY = 5



# Languages from Stage 4 output

LANGUAGES = {

    "english":
        "English",

    "tam_Taml":
        "Tamil",

    "tel_Telu":
        "Telugu",

    "mar_Deva":
        "Marathi",

    "ben_Beng":
        "Bengali"

}



# ============================================================
# GROQ INITIALIZATION
# ============================================================


client = Groq(
    api_key=GROQ_API_KEY
)


print(
    "Groq loaded:",
    MODEL_NAME
)



# ============================================================
# FILE HELPERS
# ============================================================


def load_json(path):

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)




def clean_json_response(text):

    """
    Removes markdown formatting
    and extracts JSON object.
    """


    text = text.strip()


    if "```json" in text:

        text = text.replace(
            "```json",
            ""
        )

    if "```" in text:

        text = text.replace(
            "```",
            ""
        )


    text = text.strip()


    start = text.find("{")

    end = text.rfind("}")


    if start != -1 and end != -1:

        text = text[start:end+1]


    return text




def safe_json_parse(text):

    cleaned = clean_json_response(text)

    try:
        return json.loads(cleaned)

    except Exception:

        # Try removing trailing broken JSON fragments
        while cleaned:
            try:
                return json.loads(cleaned)
            except Exception:
                cleaned = cleaned[:-1]

        return {
            "error": "JSON parsing failed",
            "raw": text[:1000]
        }




# ============================================================
# TRANSCRIPT EXTRACTION
# ============================================================


def get_transcript(data, language_code):

    """
    Extract transcript from Stage 4 JSON.
    Supports:
    - translated_text
    - segments
    - original transcript
    """


    # English

    if language_code == "english":


        if "original_transcript" in data:

            return data["original_transcript"]



        if "segments" in data:


            return " ".join(

                x.get(
                    "text",
                    ""
                )

                for x in data["segments"]

            )



    translations = data.get(
        "translations",
        {}
    )



    if language_code not in translations:

        return ""



    lang = translations[language_code]



    text = lang.get(
        "translated_text",
        ""
    )



    if text:

        return text



    segments = lang.get(
        "segments",
        []
    )


    if segments:


        return " ".join(

            x.get(
                "translated_text",
                x.get(
                    "translated_restored",
                    ""
                )
            )

            for x in segments

        )



    return ""



# ============================================================
# TEXT CHUNKING
# ============================================================


def split_text(text):

    """
    Splits transcript into safe chunks.
    """


    words = text.split()


    chunks = []

    current = []

    length = 0



    for word in words:


        length += len(word)


        current.append(word)



        if length >= MAX_CHARS_PER_CHUNK:


            chunks.append(
                " ".join(current)
            )

            current = []

            length = 0



    if current:

        chunks.append(
            " ".join(current)
        )


    return chunks


# ============================================================
# GROQ CALL FUNCTION
# ============================================================


def call_groq(prompt, retries=3):

    """
    Calls Groq API with free-tier friendly settings.
    Automatically retries rate-limit failures.
    """

    for attempt in range(retries):

        try:

            response = client.chat.completions.create(

                model=MODEL_NAME,

                messages=[

                    {
                        "role": "system",
                        "content":
                        "You are an expert educational content creator. Generate concise valid JSON notes."
                    },

                    {
                        "role": "user",
                        "content": prompt
                    }

                ],

                temperature=0.2,

                max_tokens=1200,

                response_format={
                    "type": "json_object"
                }

            )

            return response.choices[0].message.content


        except Exception as e:

            print("\nGroq error:", e)

            error_text = str(e)

            if "rate_limit" in error_text or "429" in error_text:

                wait = 60 * (attempt + 1)

                print(f"Rate limit hit. Waiting {wait}s...")

                time.sleep(wait)

            elif attempt < retries - 1:

                wait = 20 * (attempt + 1)

                print(f"Waiting {wait}s...")

                time.sleep(wait)

            else:

                raise e


    raise Exception("Groq request failed after retries")



# ============================================================
# NOTE GENERATION PROMPT
# ============================================================


def build_notes_prompt(
        transcript,
        language,
        title
):


    return f"""

You are creating university-quality lecture notes.

Lecture title:
{title}


Target language:
{language}



Create notes from the transcript below.


IMPORTANT RULES:

1. Write naturally in {language}.

2. Keep programming and technical terms in English.

Example:

Compiler (கம்பைலர்)

JVM (ஜேவிஎம்)


3. Do not translate technical keywords incorrectly.

4. Remove repeated ideas.

5. Make explanations suitable for students.


Return ONLY valid JSON. Keep explanations short and concise.


Required JSON structure:


{{

"summary":
"3-5 sentence lecture summary",


"key_concepts":

[

{{

"concept":
"important concept",

"explanation":
"clear student friendly explanation",

"example":
"example if available"

}}

],


"important_terms":

[

{{

"term":
"technical term",

"definition":
"simple definition"

}}

],



"qa_pairs":

[

{{

"question":
"student question",

"answer":
"answer"

}}

],



"quiz":

[

{{

"question":
"MCQ question",

"options":
[
"A",
"B",
"C",
"D"
],

"correct_answer":
"A",

"explanation":
"why correct"

}}

],



"learning_outcomes":

[

"learning point"

]

}}



Requirements:

- 5 key concepts
- 5 important terms
- 3 Q&A
- 3 MCQ quiz questions
- 3 learning outcomes



Transcript:

{transcript}

"""





# ============================================================
# CHUNK NOTE GENERATION
# ============================================================


def generate_notes(
        transcript,
        language,
        title
):


    chunks = split_text(
        transcript
    )


    print(
        "Chunks:",
        len(chunks)
    )



    combined = {

        "summary":
        "",


        "key_concepts":
        [],


        "important_terms":
        [],


        "qa_pairs":
        [],


        "quiz":
        [],


        "learning_outcomes":
        []

    }




    for index, chunk in enumerate(chunks):


        print(
            f"Processing chunk {index+1}/{len(chunks)}"
        )



        prompt = build_notes_prompt(

            chunk,

            language,

            title

        )



        response = call_groq(
            prompt
        )



        result = safe_json_parse(
            response
        )



        if "error" in result:

            print(
                "Invalid JSON chunk skipped"
            )

            continue




        # Merge results


        if result.get("summary"):

            combined["summary"] += (

                result["summary"]
                + " "

            )



        combined["key_concepts"].extend(

            result.get(
                "key_concepts",
                []
            )

        )



        combined["important_terms"].extend(

            result.get(
                "important_terms",
                []
            )

        )



        combined["qa_pairs"].extend(

            result.get(
                "qa_pairs",
                []
            )

        )



        combined["quiz"].extend(

            result.get(
                "quiz",
                []
            )

        )



        combined["learning_outcomes"].extend(

            result.get(
                "learning_outcomes",
                []
            )

        )



        time.sleep(
            REQUEST_DELAY
        )



    return combined



# ============================================================
# QUALITY IMPROVEMENT FUNCTIONS
# ============================================================


def remove_duplicates(items, key):

    """
    Removes duplicate objects
    based on a selected key.
    """


    seen = set()

    output = []


    for item in items:


        value = item.get(
            key,
            ""
        )


        value = value.lower().strip()


        if value not in seen:


            seen.add(value)

            output.append(item)



    return output





def clean_notes(notes):

    """
    Removes repetition and limits
    output size for educational quality.
    """



    # Remove duplicate concepts

    notes["key_concepts"] = remove_duplicates(

        notes.get(
            "key_concepts",
            []
        ),

        "concept"

    )



    # Remove duplicate terms

    notes["important_terms"] = remove_duplicates(

        notes.get(
            "important_terms",
            []
        ),

        "term"

    )



    # Remove duplicate questions

    notes["qa_pairs"] = remove_duplicates(

        notes.get(
            "qa_pairs",
            []
        ),

        "question"

    )



    # Limit size

    notes["key_concepts"] = (

        notes["key_concepts"][:5]

    )


    notes["important_terms"] = (

        notes["important_terms"][:5]

    )


    notes["qa_pairs"] = (

        notes["qa_pairs"][:10]

    )


    notes["quiz"] = (

        notes.get(
            "quiz",
            []
        )[:10]

    )


    notes["learning_outcomes"] = (

        notes.get(
            "learning_outcomes",
            []
        )[:10]

    )


    return notes





# ============================================================
# QUALITY REFINEMENT
# ============================================================


def build_refinement_prompt(
        notes,
        language
):


    return f"""

You are a professional educational editor.


Improve the following lecture notes.


Language:
{language}


Rules:

1. Remove repeated concepts.

2. Fix incorrect technical explanations.

3. Keep programming keywords in English.

4. Improve natural language quality.

5. Make notes suitable for students.

6. Do not remove important concepts.

7. Return only JSON.



Required structure:


{{

"summary":"",

"key_concepts":[],

"important_terms":[],

"qa_pairs":[],

"quiz":[],

"learning_outcomes":[]

}}



Notes:

{json.dumps(
    notes,
    ensure_ascii=False
)}

"""





def refine_notes(
        notes,
        language
):


    """
    Second AI pass for quality improvement.
    Uses smaller prompt to avoid Groq limits.
    """



    notes = clean_notes(
        notes
    )



    compact = {

        "summary":

            notes.get(
                "summary",
                ""
            )[:1500],



        "key_concepts":

            notes.get(
                "key_concepts",
                []
            )[:10],



        "important_terms":

            notes.get(
                "important_terms",
                []
            )[:10],



        "qa_pairs":

            notes.get(
                "qa_pairs",
                []
            )[:5],



        "quiz":

            notes.get(
                "quiz",
                []
            )[:5],



        "learning_outcomes":

            notes.get(
                "learning_outcomes",
                []
            )[:5]

    }



    prompt = build_refinement_prompt(

        compact,

        language

    )



    try:


        response = call_groq(
            prompt
        )


        refined = safe_json_parse(
            response
        )


        if "error" not in refined:


            return clean_notes(
                refined
            )


    except Exception as e:


        print(
            "Refinement failed:",
            e
        )



    return clean_notes(
        compact
    )

def fill_missing_fields(notes, language):

    required_fields = {
        "key_concepts": 5,
        "important_terms": 5,
        "qa_pairs": 3,
        "quiz": 3,
        "learning_outcomes": 3
    }

    missing = []

    for field, count in required_fields.items():

        if not notes.get(field) or len(notes[field]) < count:
            missing.append(field)


    if not missing:
        return notes


    prompt = f"""
You are improving educational notes.

Language:
{language}

Existing notes:
{json.dumps(notes, ensure_ascii=False)}

Missing sections:
{missing}

Generate ONLY the missing sections.

Rules:
- key_concepts: exactly 5 items
- important_terms: exactly 5 items
- qa_pairs: exactly 3 items
- quiz: exactly 3 items
- learning_outcomes: exactly 3 items

Keep answers short.
Return valid JSON only.
"""


    response = call_groq(prompt)


    try:

        extra = safe_json_parse(response)

        for field in missing:

            if field in extra:

                notes[field] = extra[field]


    except Exception:

        pass


    return notes




# ============================================================
# LANGUAGE PROCESSING
# ============================================================


def process_language(
        data,
        code,
        title
):


    transcript = get_transcript(

        data,

        code

    )


    if not transcript:


        return None



    print(
        "Characters:",
        len(transcript)
    )



    notes = generate_notes(

        transcript,

        LANGUAGES[code],

        title

    )



    print(
        "Initial notes generated"
    )



    # Remove duplicates and limit size
    refined = clean_notes(notes)


    # Fill empty or incomplete sections
    refined = fill_missing_fields(
        refined,
        LANGUAGES[code]
    )


    # Clean again after adding missing fields
    refined = clean_notes(refined)



    print(
        "Quality refinement completed"
    )



    return refined



# ============================================================
# MAIN PROGRAM
# ============================================================


def main():


    print("\nAvailable files:\n")



    files = sorted(

        OUTPUT_DIR.glob(
            "*_translated.json"
        )

    )



    if not files:

        print(
            "No translated JSON files found"
        )

        return



    for i, file in enumerate(files):

        print(
            f"[{i}] {file.name}"
        )



    print()



    while True:

        try:

            choice = int(

                input(
                    "Select file number: "
                )

            )


            if 0 <= choice < len(files):

                break


        except:


            pass



        print(
            "Invalid selection"
        )




    json_path = files[choice]



    print(

        "\nLoading:",

        json_path

    )



    data = load_json(
        json_path
    )



    title = data.get(

        "video_title",

        "Lecture"

    )



    video_id = data.get(

        "video_id",

        json_path.stem.replace(

            "_large-v3-turbo_translated",

            ""

        )

    )



    print(

        "\nTitle:",

        title

    )




    final = {


        "video_id":

            video_id,


        "video_title":

            title,


        "stage":

            "Stage 6 AI Generated Educational Notes",


        "model":

            MODEL_NAME,


        "languages":

            {}

    }




    # ========================================================
    # GENERATE ENGLISH + REGIONAL LANGUAGES
    # ========================================================



    for code, language in LANGUAGES.items():


        print(

            "\n===================="

        )


        print(

            "Generating:",

            language

        )



        try:


            result = process_language(

                data,

                code,

                title

            )



            if result is None:


                print(

                    "Language missing"

                )

                continue




            final["languages"][code] = {


                "language_name":

                    language,


                "generated_notes":

                    result

            }



            print(

                language,

                "completed"

            )



        except Exception as e:



            print(

                language,

                "FAILED:",

                e

            )



            final["languages"][code] = {


                "error":

                    str(e)

            }



        time.sleep(5)




    # ========================================================
    # SAVE OUTPUT
    # ========================================================



    output_file = OUTPUT_DIR / (

        f"{video_id}_stage6_notes.json"

    )



    with open(

        output_file,

        "w",

        encoding="utf-8"

    ) as f:



        json.dump(

            final,

            f,

            ensure_ascii=False,

            indent=2

        )




    print(

        "\n===================="

    )


    print(

        "Stage 6 completed"

    )


    print(

        "Saved:",

        output_file

    )



    print(

        "\nGenerated languages:"

    )



    for lang in final["languages"]:


        print(

            "✓",

            lang

        )





# ============================================================
# RUN
# ============================================================


if __name__ == "__main__":


    main()
