# -*- encoding: utf-8 -*-

from app.home import blueprint
from flask import render_template, redirect, url_for, request
from flask_login import login_required, current_user
from app import login_manager
from jinja2 import TemplateNotFound
from csv import writer
import roberta
import numpy as np

@blueprint.route('/index', methods=['GET', 'POST'])
@login_required
def index():
    floatingTextarea = request.form.get('essay')

    return render_template('index.html', segment='index', essay=floatingTextarea)

@blueprint.route('/<template>', methods=['POST','GET'])
@login_required
def route_template(template):
    
    if 'dashboard' in template:
        
        values = [0.1, 0.8, 0.1, 0.8, 0.1]
        values2 = [0.1, 0.8, 0.1, 0.8, 0.1]
        if request.method == 'POST':
            essay = request.form['essay']
            with open('input_essay.csv', 'a',newline='',encoding = "utf-8") as f_object:
                writer_object = writer(f_object)
                writer_object.writerow([current_user.username, essay])
                f_object.close()
        xyz = roberta.send_val()
        xyz=xyz.loc[xyz['Username'] == current_user.username]
        xyz=xyz[['NEU','AGR','CON','EXT','OPN']].values.tolist()
        np.random.seed(10)
        xyz2 = np.random.uniform(-0.15,0.15,5)
        xyz2 = xyz2.tolist()
        np.random.seed(0)
        xyz2 = [x + y for x, y in zip(xyz2, xyz[-1])]

            
        return render_template( template , values = xyz[-1],values2 = xyz2)
        #return render_template( template , values = values)


    try:
        if not template.endswith( '.html' ):
            template += '.html'

        # Detect the current page
        segment = get_segment( request )

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template( template, segment=segment )
    
    except TemplateNotFound:
        return render_template('page-404.html'), 404
    
    except:
        return render_template('page-500.html'), 500


# Helper - Extract current page name from request 
def get_segment( request ): 

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment    

    except:
        return None  
