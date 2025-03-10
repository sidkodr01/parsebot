# from flask import Flask, render_template, request, redirect, url_for, flash
# import os
# from chatbot_functions import process_pdf, process_website, process_word_document, initialize_rag_pipeline, rag_pipeline

# # Initialize Flask app
# app = Flask(__name__)
# app.secret_key = "supersecretkey"  # Required for flashing messages

# # Temporary storage for the chatbot pipeline
# qa_pipeline = None
# vectorstore = None

# @app.route("/", methods=["GET", "POST"])
# def index():
#     global qa_pipeline, vectorstore

#     if request.method == "POST":
#         # Check if a file or URL was provided
#         url_or_file = request.form.get("url_or_file")
#         file = request.files.get("file")

#         if file and file.filename:  # File upload
#             if not file.filename.endswith((".pdf", ".docx")):
#                 flash("Unsupported file type. Please upload a PDF or Word document.")
#                 return redirect(url_for("index"))

#             # Save the file temporarily
#             file_path = os.path.join("uploads", file.filename)
#             file.save(file_path)

#             # Process the file based on its type
#             if file.filename.endswith(".pdf"):
#                 texts = process_pdf(file_path)
#             else:
#                 texts = process_word_document(file_path)

#         elif url_or_file:  # URL input
#             if not url_or_file.startswith(("http://", "https://")):
#                 flash("Please enter a valid URL.")
#                 return redirect(url_for("index"))

#             # Process the website
#             texts = process_website(url_or_file)

#         else:
#             flash("No input provided. Please enter a URL or upload a file.")
#             return redirect(url_for("index"))

#         # Initialize the RAG pipeline
#         qa_pipeline, vectorstore = initialize_rag_pipeline(texts)
#         flash("Document processed successfully. You can now ask questions.")
#         return redirect(url_for("index"))

#     return render_template("index.html")

# @app.route("/ask", methods=["POST"])
# def ask():
#     global qa_pipeline, vectorstore

#     if not qa_pipeline or not vectorstore:
#         flash("Please upload a document or enter a URL first.")
#         return redirect(url_for("index"))

#     user_query = request.form["query"]
#     if not user_query:
#         flash("Please enter a question.")
#         return redirect(url_for("index"))

#     # Get the response from the RAG pipeline
#     response = rag_pipeline(user_query, qa_pipeline, vectorstore)
#     return render_template("index.html", query=user_query, response=response)

# if __name__ == "__main__":
#     # Create uploads directory if it doesn't exist
#     if not os.path.exists("uploads"):
#         os.makedirs("uploads")

#     # Run the Flask app
#     app.run(debug=True)


import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for, flash
from chatbot_functions import process_pdf, process_website, process_word_document, initialize_rag_pipeline, rag_pipeline

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for flashing messages

# Temporary storage for the chatbot pipeline
qa_pipeline = None
vectorstore = None

# Create a temporary directory for file uploads
UPLOAD_FOLDER = tempfile.mkdtemp()

@app.route("/", methods=["GET", "POST"])
def index():
    global qa_pipeline, vectorstore

    if request.method == "POST":
        # Check if a file or URL was provided
        url_or_file = request.form.get("url_or_file")
        file = request.files.get("file")

        if file and file.filename:  # File upload
            if not file.filename.endswith((".pdf", ".docx")):
                flash("Unsupported file type. Please upload a PDF or Word document.")
                return redirect(url_for("index"))

            # Save the file temporarily
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            # Process the file based on its type
            if file.filename.endswith(".pdf"):
                texts = process_pdf(file_path)
            else:
                texts = process_word_document(file_path)

            # Initialize the RAG pipeline
            qa_pipeline, vectorstore = initialize_rag_pipeline(texts)
            flash("Document processed successfully. You can now ask questions.")

            # Delete the file after processing
            os.remove(file_path)

        elif url_or_file:  # URL input
            if not url_or_file.startswith(("http://", "https://")):
                flash("Please enter a valid URL.")
                return redirect(url_for("index"))

            # Process the website
            texts = process_website(url_or_file)

            # Initialize the RAG pipeline
            qa_pipeline, vectorstore = initialize_rag_pipeline(texts)
            flash("Document processed successfully. You can now ask questions.")

        else:
            flash("No input provided. Please enter a URL or upload a file.")
            return redirect(url_for("index"))

    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    global qa_pipeline, vectorstore

    if not qa_pipeline or not vectorstore:
        flash("Please upload a document or enter a URL first.")
        return redirect(url_for("index"))

    user_query = request.form["query"]
    if not user_query:
        flash("Please enter a question.")
        return redirect(url_for("index"))

    # Get the response from the RAG pipeline
    response = rag_pipeline(user_query, qa_pipeline, vectorstore)
    return render_template("index.html", query=user_query, response=response)

@app.route("/clear", methods=["POST"])
def clear():
    global qa_pipeline, vectorstore
    qa_pipeline = None
    vectorstore = None
    flash("Session cleared. Please upload a new document or enter a URL.")
    return redirect(url_for("index"))

if __name__ == "__main__":
    # Run the Flask app
    app.run(debug=True)