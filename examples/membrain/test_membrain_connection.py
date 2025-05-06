#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple test to verify Membrain connection works
"""
import asyncio
import aiohttp
import time

async def main():
    # Configuration
    membrain_endpoint = "http://localhost:9201"
    membrain_namespace = "test"
    key = "test_key"
    value = b"Hello, Membrain!"

    # Create session
    timeout = aiohttp.ClientTimeout(total=5.0)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Health check
        try:
            print(f"Checking health at {membrain_endpoint}/healthz...")
            async with session.get(f"{membrain_endpoint}/healthz") as response:
                if response.status == 200:
                    print("Health check OK!")
                else:
                    print(f"Health check failed with status {response.status}")
                    return
        except Exception as e:
            print(f"Health check failed with error: {e}")
            return

        # Try to store a small value
        try:
            url = f"{membrain_endpoint}/memory/{membrain_namespace}/{key}"
            print(f"Attempting to PUT data to {url}...")
            start = time.time()
            async with session.put(url, data=value) as response:
                took = time.time() - start
                if response.status == 200:
                    print(f"PUT succeeded in {took:.3f}s")
                else:
                    print(f"PUT failed with status {response.status} in {took:.3f}s")
                    print(await response.text())
        except Exception as e:
            print(f"PUT failed with error: {e}")
            
        # Try to get the value back
        try:
            url = f"{membrain_endpoint}/memory/{membrain_namespace}/{key}"
            print(f"Attempting to GET data from {url}...")
            start = time.time()
            async with session.get(url) as response:
                took = time.time() - start
                if response.status == 200:
                    data = await response.read()
                    print(f"GET succeeded in {took:.3f}s, data: {data}")
                else:
                    print(f"GET failed with status {response.status} in {took:.3f}s")
                    print(await response.text())
        except Exception as e:
            print(f"GET failed with error: {e}")

if __name__ == "__main__":
    asyncio.run(main())