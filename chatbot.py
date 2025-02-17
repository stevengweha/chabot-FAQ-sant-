###creation de l'API flask pour tester nos données via postman
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Charger le modèle préalablement sauvegardé
model = joblib.load("chatbot_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données JSON envoyées dans la requête
        data = request.get_json()

        # Vérifier si 'features' est présent dans la requête
        if 'features' not in data:
            return jsonify({"error": "No features found in request"}), 400

        # Extraire les caractéristiques envoyées dans le JSON
        features = data['features']

        # Effectuer la prédiction avec le modèle
        prediction = model.predict([features])
        disease_type = prediction[0]  # La maladie prédite par le modèle

        # Vous pouvez également obtenir les probabilités des classes
        prediction_probabilities = model.predict_proba([features])

        # Renvoyer la maladie exacte ainsi que les probabilités (si nécessaire)
        return jsonify({
            "prediction": disease_type,
            "probabilities": prediction_probabilities[0].tolist()  # Probabilités pour chaque classe
        })

    except Exception as e:
        # Gérer les erreurs
        return jsonify({"error": str(e)}), 500

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
    # Lancer le serveur Flask
    Timer(10, stop_flask).start()  # Arrêter après 10 secondes
    app.run(debug=True, use_reloader=False)  # use_reloader=False pour éviter que Flask ne r
