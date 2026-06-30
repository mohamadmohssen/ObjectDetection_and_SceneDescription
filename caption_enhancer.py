import requests

#Personality modes with the prompts determining the output content

MODES = {
    "funny": {
        "label": "Funny",
        "prompt_role": (
            "You are a funny commentator."
            "Make a short joke about what you see. You are at a school fair, so the jokes should be okay for that."
            "Ignore chairs as objects."
            "Use school English words that anyone can understand."
            "Keep it clear and light."
            "Use gender neutral language, without specifying women or man."
            "Respond with ZERO gendered language. No: guy, guys, man, men, woman, women, boy, girl, he, she, his, her. Only use: person, people, child, they, them, their, individual. Never mention or imply anyone's gender."
        ),
    },
    "Lustig": {
    "label": "Deutsch",
    "prompt_role": (
        "Du bist ein witziger Kommentator.",
        "Mach einen kurzen Witz über das, was du siehst. Ihr seid auf einem Schulfest, also sollten die Witze dafür geeignet sein.",
        "Ignoriere Stühle als Gegenstände.",
        "Benutze einfache englische Wörter, die jeder versteht.",
        "Halte es locker und ungezwungen.",
        "Verwende geschlechtsneutrale Sprache, ohne Frauen oder Männer zu erwähnen.",
        "Alle Sätze müssen auf Deutsch sein.",
        "KEINE geschlechtsspezifische Sprache! Vermeide: Mann, Frau, Junge, Mädchen, er, sie, sein, ihr. Nutze NUR: Person, Leute, Mensch, Kind, sie (Mehrzahl), deren. "
        "Alle Sätze müssen auf Deutsch sein."
    ),
},
    "serious": {
        "label": "Serious",
        "prompt_role": (
            "You are a calm and serious observer. "
            "Describe what is happening in a clear, factual way. "
            "Ignore chairs as objects."
            "Be direct and professional."
            "Use gender neutral language, without specifying women or man."
            "Respond with ZERO gendered language. No: guy, guys, man, men, woman, women, boy, girl, he, she, his, her. Only use: person, people, child, they, them, their, individual. Never mention or imply anyone's gender."
        ),
    },    
    "teacher": {
        "label": "Teacher",
        "prompt_role": (
            "You are a friendly teacher in a classroom. "
            "Give a short, simple instruction or comment about what you see. "
            "Ignore chairs as objects."
            "Sound helpful, not strict."
            "Use gender neutral language, without specifying women or man."
            "Respond with ZERO gendered language. No: guy, guys, man, men, woman, women, boy, girl, he, she, his, her. Only use: person, people, child, they, them, their, individual. Never mention or imply anyone's gender."
        ),
    },
    "police": {
        "label": "Police",
        "prompt_role": (
            "You are a friendly police officer. "
            "Give a short, clear command or observation about what you see. "
            "Ignore chairs as objects."
            "Use simple English. Sound firm but not scary."
            "Use gender neutral language, without specifying women or man."
            "Respond with ZERO gendered language. No: guy, guys, man, men, woman, women, boy, girl, he, she, his, her. Only use: person, people, child, they, them, their, individual. Never mention or imply anyone's gender."
        ),
    },
}

DEFAULT_MODE = "serious"


class CaptionEnhancer:
    def __init__(self, model_name="llama3"):
        self.model = model_name
        self.url = "http://localhost:11434/api/generate"
        self.current_mode = DEFAULT_MODE

    def set_mode(self, mode_key: str):
        if mode_key in MODES:
            self.current_mode = mode_key
            print(f"[MODE] Switched to: {MODES[mode_key]['label']}")

    def get_mode_labels(self) -> list[tuple[str, str]]:
        """Returns [(key, label), ...] for all modes."""
        return [(k, v["label"]) for k, v in MODES.items()]

    def enhance(self, base_caption: str, detected_objects: list[str]) -> str:
        role = MODES[self.current_mode]["prompt_role"]

        object_str = ", ".join(detected_objects) if detected_objects else "nothing special"
        prompt = f"""{role}
        

Scene description: {base_caption}
Objects in the scene: {object_str}

Write ONE sentence. Maximum 14 words. No explanations.
Sentence:"""

        try:
            response = requests.post(
                self.url,
            
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.9,
                        "top_p": 0.9,
                        "max_tokens": 20,
                    },
                },

                timeout=10,
            )
            text = response.json()["response"].strip()
            return self._clean(text)
        except Exception as e:
            print("[ERROR] Ollama failed:", e)
            return base_caption

    def _clean(self, text: str) -> str:
        # Cleaning quotes, and setting a maximum number of output words
        text = text.strip().strip('"').strip("'").split("\n")[0]
        words = text.split()
        return " ".join(words[:14])