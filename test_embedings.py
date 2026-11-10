import requests

API_KEY = "AQ.Ab8RN6KFnDRuUSjk5ZA3WdBzS7UqkV6veGkgwlm5ejZvFI1tOg"

def vezi_modele_vectorizare():
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"
    raspuns = requests.get(url)

    if raspuns.status_code == 200:
        modele = raspuns.json().get('models', [])
        print("Embeddings disponibile:\n")

        for model in modele:

            if 'embedContent' in model.get('supportedGenerationMethods', []):
                print(f"- {model['name']}")
    else:
        print(" Eroare la conectare")

if __name__ == "__main__":
    vezi_modele_vectorizare()