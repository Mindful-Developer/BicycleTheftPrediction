import json

def refactor_geodata():
    with open('toronto_map_data_raw.json', 'r') as f:
        data = json.load(f)

    extracted = {'regions': []}
    for region in data['features']:
        region_data = {
            'NEIGHBOURHOOD_140': region['attributes']['AREA_DESC'],
            'HOOD_140': region['attributes']['AREA_LONG_CODE'],
            'geometry': region['geometry']['rings'][0]
        }
        extracted['regions'].append(region_data)

    with open('../api/toronto_map_data_extracted.json', 'w') as f:
        json.dump(extracted, f, indent=4)


refactor_geodata()