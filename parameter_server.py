import time
from typing import Any

import requests  # type: ignore


def make_post_request(request_id: Any) -> None:
    """Makes a single POST request to the specified endpoint."""
    url = "http://localhost:8082/v1/chimera-parameter-server/fit"
    try:
        start_time = time.time()
        response = requests.post(url)  # POST request
        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            print(
                f"Request {request_id}: Success - Status Code: {response.status_code}, Response Time: {response_time:.4f} seconds"
            )
            # You can process the response.json() here if needed
            # data = response.json()
            # print(f"Request {request_id}: Response Data: {data}")
        else:
            print(
                f"Request {request_id}: Failed - Status Code: {response.status_code}, Response Time: {response_time:.4f} seconds, Response Text: {response.text}"
            )

    except requests.exceptions.ConnectionError as e:
        print(
            f"Request {request_id}: Connection Error - Could not connect to {url}: {e}"
        )
    except requests.exceptions.RequestException as e:
        print(
            f"Request {request_id}: Request Error - An error occurred during the request: {e}"
        )
    except Exception as e:
        print(f"Request {request_id}: An unexpected error occurred: {e}")


def main() -> None:
    """Runs 30 POST requests to the specified endpoint sequentially."""
    num_requests = 30

    print(f"Starting {num_requests} POST requests...")

    for i in range(num_requests):
        make_post_request(i + 1)

    print("All requests completed.")


if __name__ == "__main__":
    main()
