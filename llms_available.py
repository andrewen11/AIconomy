import os
import requests

API_KEY = "AQ.Ab8RN6KFnDRuUSjk5ZA3WdBzS7UqkV6veGkgwlm5ejZvFI1tOg"

def vezi_modele_disponibile():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    raspuns = requests.get(url)

    if raspuns.status_code == 200:
        modele = raspuns.json().get('models', [])
        print("Modele disponibile - generare text (CrewAI):\n")

        for model in modele:

            metode_suportate = model.get('supportedGenerationMethods', [])
            if 'generateContent' in metode_suportate:
                print(f"- {model['name']}")
    else:
        print(f"Eroare la conectare: {raspuns.status_code}")
        print(raspuns.text)

if __name__ == "__main__":
    vezi_modele_disponibile()