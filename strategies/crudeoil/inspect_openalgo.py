import sys
import inspect
import os

try:
    from openalgo import api
    
    methods = [
        '__getattr__',
        '_handle_response'
    ]
    
    with open("api_source_3.txt", "w", encoding="utf-8") as f:
        f.write("--- API Inspection 3 ---\n")
        
        for m in methods:
            if hasattr(api, m):
                method = getattr(api, m)
                f.write(f"\nMethod: {m}\n")
                try:
                    src = inspect.getsource(method)
                    f.write(src)
                except Exception as e:
                    f.write(f"Could not get source: {e}\n")
                    f.write(f"Doc: {method.__doc__}\n")
            else:
                f.write(f"\nMethod {m} NOT FOUND in api class\n")

except Exception as e:
    with open("api_source_3.txt", "w") as f:
        f.write(f"Error: {e}")
