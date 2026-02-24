from flask import Flask, render_template, request
from src.predict import predict_email

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None

    if request.method == "POST":
        email_text = request.form.get("email")

        if email_text:
            result, confidence = predict_email(email_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
