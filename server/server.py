from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import math

app = Flask(__name__)
CORS(app)

months = ['Oct', 'Mar', 'Feb', 'May', 'Nov', 'Dec', 'Apr', 'Aug', 'Jan', 'Sep', 'Jul', 'Jun']
counties = ['GARFIELD', 'WALLA WALLA', 'GRAYS HARBOR', 'FERRY', 'OKANOGAN', 'LINCOLN', 'STEVENS', 'PIERCE', 'COWLITZ', 'YAKIMA', 'PACIFIC', 'ASOTIN', 'CHELAN', 'DOUGLAS', 'WHITMAN', 'SKAMANIA', 'GRANT', 'KLICKITAT', 'KING', 'MASON', 'FRANKLIN', 'JEFFERSON', 'KITTITAS', 'ADAMS', 'SAN JUAN', 'SPOKANE', 'BENTON', 'ISLAND', 'KITSAP', 'NO DATA', 'WHATCOM', 'THURSTON', 'CLALLAM', 'SKAGIT', 'LEWIS', 'WAHKIAKUM', 'SNOHOMISH', 'COLUMBIA', 'CLARK', 'PEND OREILLE']
causes = ['Railroad', 'Arson', 'Recreation', 'Under Invest', 'Logging', 'Children', 'Lightning', 'Debris Burn', 'Miscellaneou', 'None', 'Undetermined', 'Smoker']
binlat = [1, 2, 3, 4]
binlon = [1, 2, 3, 4, 5, 6, 7, 8]
binacres = [2, 3, 4, 5, 6, 7, 8, 9]


def acres_to_circle_radius_in_miles(acres):
    sqft = acres * 43560
    radius = math.sqrt(sqft / math.pi)
    return radius / 5280

@app.route('/', methods=['GET'])
def main_route():
     return render_template('index.html')

@app.route('/api/predict')
def return_prediction():
    acres = 10000
    
    result = {
        "acres": acres,
        "radius_miles": acres_to_circle_radius_in_miles(acres) 
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)