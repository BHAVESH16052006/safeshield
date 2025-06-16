import nltk
import ssl
import os

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Set NLTK data path to the current directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

print("Downloading NLTK data...")
resources = ['punkt', 'stopwords', 'punkt_tab', 'averaged_perceptron_tagger']
for resource in resources:
    try:
        nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
        print(f"Downloaded {resource} successfully!")
    except Exception as e:
        print(f"Warning while downloading {resource}: {str(e)}")

print("NLTK setup completed!") 