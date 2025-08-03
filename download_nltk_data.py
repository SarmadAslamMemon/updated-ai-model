import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("📥 Downloading required NLTK data...")

# Download required NLTK data
try:
    nltk.download('punkt')
    print("✅ Downloaded punkt")
except Exception as e:
    print(f"⚠️ Error downloading punkt: {e}")

try:
    nltk.download('stopwords')
    print("✅ Downloaded stopwords")
except Exception as e:
    print(f"⚠️ Error downloading stopwords: {e}")

try:
    nltk.download('wordnet')
    print("✅ Downloaded wordnet")
except Exception as e:
    print(f"⚠️ Error downloading wordnet: {e}")

try:
    nltk.download('averaged_perceptron_tagger')
    print("✅ Downloaded averaged_perceptron_tagger")
except Exception as e:
    print(f"⚠️ Error downloading averaged_perceptron_tagger: {e}")

print("🎉 NLTK data download complete!") 