import tempfile

from langchain_community.document_loaders import PDFPlumberLoader


def load_legal_document(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PDFPlumberLoader(tmp_path)
    return loader.load()
