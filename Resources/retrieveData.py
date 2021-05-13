import requests


def download_file_from_google_drive(id):
    destination = 'data/'+id
    if id == 'allNoisyData.csv':
        URL = "https://drive.google.com/file/d/1BHe7DX0Jz8xbwNtRr5B_wlzcDCXnCW62/view?usp=sharing"
    else:
        URL = "https://drive.google.com/u/1/uc?id=16Zjvc0iqcWAznTCIceG_i3CBoGMuJfYt&export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
