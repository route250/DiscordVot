
import httpx
import asyncio
from typing import List, Optional, Set

async def a_find_first_responsive_host(hostname_list: List[str], port: Optional[int] = None, timeout: float = 1.0) -> Optional[str]:
    uniq: Set[str] = set()
    async with httpx.AsyncClient(timeout=timeout) as client:
        for sv in hostname_list:
            url = f"{sv}"
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url
            if port is not None:
                url += f":{port}"
            
            if url not in uniq:
                uniq.add(url)
                try:
                    response = await client.get(url)
                    if response.status_code in {200, 404}:
                        return url
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                    continue

    return None

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