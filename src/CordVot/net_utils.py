
import requests

def find_first_responsive_host(hostname_list:list[str], port:int|None=None, timeout:float=1.0) ->str|None:
    uniq:set = set()
    for sv in hostname_list:
        url = f"{sv}"
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://"+url
        if port is not None:
            url += f":{port}"
        if url not in uniq:
            uniq.add(url)
            try:
                response = requests.get(url, timeout=timeout)
                if response.status_code == 200 or response.status_code == 404:
                    return url
            except (requests.ConnectionError, requests.Timeout):
                continue

    return None