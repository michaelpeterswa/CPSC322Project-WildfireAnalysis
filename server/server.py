
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import math
import pickle

app = Flask(__name__)
CORS(app)

months = ['Oct', 'Mar', 'Feb', 'May', 'Nov', 'Dec', 'Apr', 'Aug', 'Jan', 'Sep', 'Jul', 'Jun']
counties = ['GARFIELD', 'WALLA WALLA', 'GRAYS HARBOR', 'FERRY', 'OKANOGAN', 'LINCOLN', 'STEVENS', 'PIERCE', 'COWLITZ', 'YAKIMA', 'PACIFIC', 'ASOTIN', 'CHELAN', 'DOUGLAS', 'WHITMAN', 'SKAMANIA', 'GRANT', 'KLICKITAT', 'KING', 'MASON', 'FRANKLIN', 'JEFFERSON', 'KITTITAS', 'ADAMS', 'SAN JUAN', 'SPOKANE', 'BENTON', 'ISLAND', 'KITSAP', 'NO DATA', 'WHATCOM', 'THURSTON', 'CLALLAM', 'SKAGIT', 'LEWIS', 'WAHKIAKUM', 'SNOHOMISH', 'COLUMBIA', 'CLARK', 'PEND OREILLE']
causes = ['Railroad', 'Arson', 'Recreation', 'Under Invest', 'Logging', 'Children', 'Lightning', 'Debris Burn', 'Miscellaneou', 'None', 'Undetermined', 'Smoker']
binlat = [1, 2, 3, 4]
binlon = [1, 2, 3, 4, 5, 6, 7, 8]
binacres = [2, 3, 4, 5, 6, 7, 8, 9]

def binLat(lat):
    print(lat)
    if lat > 48:
        return 1
    elif 48 >= lat > 47:
        return 2
    elif 47 >= lat > 46:
        return 3
    elif 46 >= lat > 45:
        return 4
    else:
        return 5

def binLon(lon):
    if lon < -124:
        return 1
    elif -124 <= lon < -123:
        return 2
    elif -123 <= lon < -122:
        return 3
    elif -122 <= lon < -121:
        return 4
    elif -121 <= lon < -120:
        return 5
    elif -120 <= lon < -119:
        return 6
    elif -119 <= lon < -118:
        return 7
    else:
        return 8

def unBinAcres(acres_binned):
    if acres_binned == 1:
        return "0-2"
    if acres_binned == 2:
        return "2-10"
    if acres_binned == 3:
        return "10-50"
    if acres_binned == 4:
        return "50-100"
    if acres_binned == 5:
        return "100-500"
    if acres_binned == 6:
        return "500-2000"
    if acres_binned == 7:
        return "2000-10000"
    if acres_binned == 8:
        return "10000-50000"
    return "50000-"

def acres_to_circle_radius_in_miles(acres):
    sqft = acres * 43560
    radius = math.sqrt(sqft / math.pi)
    return radius / 5280

@app.route('/', methods=['GET'])
def main_route():
     return render_template('index.html')

@app.route('/api/predict', methods=["GET"])
def return_prediction():
    acres = 10000

    cause = request.args.get("cause", "")
    county = request.args.get("county", "")
    fire_date = request.args.get("date", "")
    lat = request.args.get("lat", "")
    lon = request.args.get("lon", "")

    lat = binLat(float(lat))
    lon = binLon(float(lon))

    instance = [fire_date, county, cause, lat, lon]

    infile = open("trees.p", "rb")
    best_trees = pickle.load(infile)
    infile.close()

    prediction = predict_acres([instance], best_trees)
    
    if prediction is not None:
        acres_binned = prediction[0]
        result = {"prediction": unBinAcres(acres_binned)}
        return jsonify(result), 200
    else: 
        # failure!!
        return "Error making prediction", 400



def predict_acres(X_test, best_trees):
    header = []
    predictions = []
    for i in range(0, len(X_test[0])):
        header.append("att" + str(i))
    for instance in X_test:
        tree_predictions = {}
        for tree in best_trees:
            temp = tree['tree']
            prediction = tdidt_predict(header, tree['tree'], instance)
            if prediction in tree_predictions:
                tree_predictions[prediction] += 1
            else:
                tree_predictions[prediction] = 1
        
        max_key = max(tree_predictions, key = tree_predictions.get)
        predictions.append(max_key)
    return predictions

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        # now I need to find which "edge" to follow recursively
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                # we have a match!! recurse!!
                return tdidt_predict(header, value_list[2], instance)
    else: # "Leaf"
        return tree[1] # leaf class label


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888)