import requests
import json
import pytest

def test_health():
    """Test the health check endpoint."""
    url = "http://localhost:5000/health"
    try:
        response = requests.get(url)
        print("\nHealth Check Response:")
        print(json.dumps(response.json(), indent=2))
        assert response.status_code == 200
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error connecting to server: {e}")

def test_prediction():
    """Test the prediction endpoint with sample data."""
    url = "http://localhost:5000/predict"

    sample_data = {
        "BIKE_MAKE": "TREK",
        "BIKE_MODEL": "FX3",
        "BIKE_TYPE": "REGULAR",
        "BIKE_SPEED": 21,
        "BIKE_COLOUR": "BLACK",
        "BIKE_COST": 800,
        "STATUS": "STOLEN",
        "PREMISES_TYPE": "OUTDOOR",
        "LOCATION_TYPE": "STREET",
        "OCC_DATE": "2023-01-01",
        "OCC_DOW": "Sunday",
        "OCC_HOUR": 14,
        "REPORT_DATE": "2023-01-02",
        "HOOD_140": "Bay Street Corridor",
        "NEIGHBOURHOOD_140": "Downtown",
        "OCC_DOY": 1
    }

    try:
        response = requests.post(url, json=sample_data)
        print("\nAPI Response:")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        assert response.status_code == 200, f"API returned status code {response.status_code}"
            
        print("\nPrediction Details:")
        print(f"Status: {result['status']}")
        print(f"Recovery Probability: {result['probability_recovered']:.2%}")
        print(f"Stolen Probability: {result['probability_stolen']:.2%}")
        
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Error connecting to server: {e}")
    except Exception as e:
        pytest.fail(f"Error processing response: {e}")

def test_minimal_input():
    """Test prediction with minimal required fields."""
    url = "http://localhost:5000/predict"
    minimal_data = {
        "BIKE_TYPE": "REGULAR",
        "PREMISES_TYPE": "OUTDOOR",
        "LOCATION_TYPE": "STREET",
        "OCC_DATE": "2023-01-01",
        "REPORT_DATE": "2023-01-02",
        "OCC_HOUR": 14,
        "OCC_DOW": "Sunday",
        "OCC_DOY": 1,
        "HOOD_140": "Bay Street Corridor",
        "NEIGHBOURHOOD_140": "Downtown",
        "BIKE_SPEED": 0,
        "BIKE_COST": 0
    }
    
    try:
        response = requests.post(url, json=minimal_data)
        print(f"\nMinimal Input Response: {response.text}")
        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
        result = response.json()
        assert 'status' in result
        assert 'probability_recovered' in result
        assert 'probability_stolen' in result
        assert isinstance(result['probability_recovered'], float)
        assert isinstance(result['probability_stolen'], float)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON response: {str(e)}")
    except AssertionError as e:
        pytest.fail(f"Assertion failed: {str(e)}\nResponse: {response.text}")

def test_missing_optional_fields():
    """Test prediction with some optional fields missing."""
    url = "http://localhost:5000/predict"
    partial_data = {
        "BIKE_TYPE": "REGULAR",
        "PREMISES_TYPE": "OUTDOOR",
        "LOCATION_TYPE": "STREET",
        "OCC_DATE": "2023-01-01",
        "REPORT_DATE": "2023-01-02",
        "OCC_HOUR": 14,
        "OCC_DOW": "Sunday",
        "OCC_DOY": 1,
        "HOOD_140": "Bay Street Corridor",
        "NEIGHBOURHOOD_140": "Downtown",
        "BIKE_SPEED": 0,
        "BIKE_COST": 800,

        # Optional fields included with some missing
        "BIKE_COLOUR": "BLACK"
    }
    
    try:
        response = requests.post(url, json=partial_data)
        print(f"\nPartial Data Response: {response.text}")
        assert response.status_code == 200, f"Request failed with status {response.status_code}: {response.text}"
        result = response.json()
        assert 'status' in result
        assert isinstance(result['probability_recovered'], float)
        assert isinstance(result['probability_stolen'], float)
    except requests.exceptions.RequestException as e:
        pytest.fail(f"Request failed: {str(e)}")
    except json.JSONDecodeError as e:
        pytest.fail(f"Invalid JSON response: {str(e)}")
    except AssertionError as e:
        pytest.fail(f"Assertion failed: {str(e)}\nResponse: {response.text}")

def test_invalid_data_types():
    """Test prediction with invalid data types."""
    url = "http://localhost:5000/predict"
    invalid_data = {
        "BIKE_TYPE": 123,  # Should be string
        "BIKE_SPEED": "fast",  # Should be integer
        "OCC_DATE": "invalid-date",
        "HOOD_140": ["invalid", "type"],
        "OCC_HOUR": "noon",  # Should be integer
        "OCC_DOY": "first-day",  # Should be integer
        "BIKE_COST": "expensive"  # Should be number
    }
    
    response = requests.post(url, json=invalid_data)
    assert response.status_code == 400
    result = response.json()
    assert 'error' in result

def test_empty_request():
    """Test prediction with empty request body."""
    url = "http://localhost:5000/predict"
    response = requests.post(url, json={})
    assert response.status_code == 400
    result = response.json()
    assert 'error' in result

def test_missing_critical_fields():
    """Test prediction with missing critical fields."""
    url = "http://localhost:5000/predict"
    incomplete_data = {
        "BIKE_MAKE": "TREK",
        "BIKE_MODEL": "FX3",
        "BIKE_COLOUR": "BLACK",
        # Missing critical fields: OCC_DATE, LOCATION_TYPE, etc.
    }
    
    response = requests.post(url, json=incomplete_data)
    assert response.status_code == 400
    result = response.json()
    assert 'error' in result

if __name__ == "__main__":
    print("Testing Bicycle Theft Prediction API...")
    
    # Run all tests
    test_functions = [
        test_health,
        test_prediction,
        test_minimal_input,
        test_missing_optional_fields,
        test_invalid_data_types,
        test_empty_request,
        test_missing_critical_fields
    ]
    
    failed_tests = []
    for test_func in test_functions:
        print(f"\nRunning {test_func.__name__}...")
        try:
            test_func()
        except AssertionError as e:
            print(f"Test failed with assertion error: {e}")
            failed_tests.append(test_func.__name__)
        except Exception as e:
            print(f"Test failed with error: {e}")
            failed_tests.append(test_func.__name__)
    
    if failed_tests:
        print(f"\nThe following tests failed: {', '.join(failed_tests)}")
        exit(1)
    else:
        print("\nAll tests passed successfully!")
