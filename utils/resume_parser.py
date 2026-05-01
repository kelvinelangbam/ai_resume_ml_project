import PyPDF2

def extract_text(file):
    text = ""

    try:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print("Error:", e)
        return ""

    text = text.replace("\n", " ")
    return text.strip()