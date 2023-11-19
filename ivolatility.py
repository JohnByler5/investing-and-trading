import pandas as pd
import requests
import time
from io import BytesIO


class AuthError(Exception):
    pass


__username__ = None
__password__ = None
__token__ = None
__auth__ = None

rest_api_url = 'https://restapi.ivolatility.com'


class Auth(requests.auth.AuthBase):
    def __init__(self, api_key):
        self.api_key = api_key

    def __call__(self, r):
        r.headers["apiKey"] = self.api_key
        return r


def set_rest_api_url(url):
    global rest_api_url
    rest_api_url = url


def get_token(username, password):
    return requests.get(rest_api_url + '/token/get', params={'username': username, 'password': password}).text


def create_api_key(name_key, username, password):
    return requests.post(rest_api_url + '/keys?nameKey={}'.format(name_key),
                         json={'username': username, 'password': password}).json()['key']


def delete_api_key(name_key, username, password):
    return requests.delete(rest_api_url + '/keys?nameKey={}'.format(name_key),
                           json={'username': username, 'password': password}).status_code == 200


def set_login_params(username=None, password=None, token=None, api_key=None):
    global __username__
    global __password__
    global __token__
    global __auth__

    __username__ = username
    __password__ = password
    __token__ = token
    __auth__ = None
    if api_key is not None:
        __auth__ = Auth(api_key)


def set_method(endpoint):
    login_params = {}
    if __auth__ is not None:
        pass
    elif __token__ is not None:
        login_params = {'token': __token__}
    elif __username__ is not None and __password__ is not None:
        login_params = {'username': __username__, 'password': __password__}

    url = rest_api_url + endpoint

    def get_market_data_from_file(url_for_details, pause=1, timeout=10):
        start = time.time()
        while time.time() - start < timeout:

            try:
                response = requests.get(url_for_details, auth=__auth__, timeout=timeout).json()
                if response[0]['meta']['status'] == 'COMPLETE' and response[0]['data'][0]['urlForDownload']:
                    break
            except (IndexError, TimeoutError, requests.exceptions.RequestException):
                time.sleep(pause)

        else:
            return 'Reached timeout for fetching download URL', pd.DataFrame()

        start = time.time()
        while time.time() - start < timeout:

            try:
                file_response = requests.get(response[0]['data'][0]['urlForDownload'], auth=__auth__, timeout=timeout)
            except (TimeoutError, requests.exceptions.RequestException):
                time.sleep(pause)
                continue

            if file_response.status_code != 200:
                return f'{file_response.status_code} {file_response.text}', pd.DataFrame()

            market_data = pd.read_csv(BytesIO(file_response.content), compression='gzip')
            if not market_data.empty:
                return None, market_data

            time.sleep(pause)

        return 'Reached timeout for downloading data file', pd.DataFrame()

    def request_market_data(timeout=10, pause=1, **params):
        start = time.time()
        while time.time() - start < timeout:
            try:
                req = requests.get(url, auth=__auth__, params=params, timeout=timeout)
                break
            except requests.RequestException:
                continue
        else:
            return 'Reached timeout for initial data fetch', pd.DataFrame()

        if req.status_code in [200, 400]:
            if endpoint in ['/quotes/options', '/equities/rtdl/options-rawiv']:
                return None, pd.read_csv(BytesIO(req.content))

            if endpoint in ['/options/rt-equity-nbbo']:
                return None, pd.read_csv(BytesIO(req.content), compression='zip')

            try:
                req_json = req.json()
            except:
                return 'JSON failed', pd.DataFrame()

            exception_endpoints = ['/proxy/option-series', '/futures/prices/options', '/futures/market-structure',
                                   '/equities/option-series', '/futures/rt/single-fut-opt-rawiv',
                                   '/futures/fut-opt-market-structure', '/equities/eod/option-series-on-date']
            if endpoint in exception_endpoints:
                return None, pd.DataFrame(req_json)
            elif 'status' in req_json and req_json['status']['code'] == 'PENDING':
                return get_market_data_from_file(req_json['status']['urlForDetails'], pause=pause, timeout=timeout)
            else:
                try:
                    return None, pd.DataFrame(req_json['data'])
                except:
                    return 'DataFrame creation failed', pd.DataFrame()

        elif req.status_code == 401:
            raise AuthError(f"{req.status_code} Invalid password for username {__username__}")

        else:
            try:
                text = eval(req.text)["error"]
            except:
                text = req.text
            return f'{req.status_code} {text}', pd.DataFrame()

    def factory(**kwargs):
        params = dict(login_params, **kwargs)
        if 'from_' in params.keys():
            params['from'] = params.pop('from_')
        elif '_from' in params.keys():
            params['from'] = params.pop('_from')
        elif '_from_' in params.keys():
            params['from'] = params.pop('_from_')
        return request_market_data(**params)

    return factory
