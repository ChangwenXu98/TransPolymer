import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

from sklearn.manifold import TSNE

from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.PandasTools import ChangeMoleculeRendering

#Bokeh library for plotting
import json
from bokeh.plotting import figure, show, output_notebook, ColumnDataSource
from bokeh.models import HoverTool
from bokeh.transform import factor_cmap
from bokeh.plotting import figure, output_file, save
output_notebook()

import pdb

def _prepareMol(mol,kekulize):
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    return mc

def moltosvg(mol,molSize=(450,200),kekulize=True,drawer=None,**kwargs):
    mc = _prepareMol(mol,kekulize)
    if drawer is None:
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc,**kwargs)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return SVG(svg.replace('svg:',''))

df = pd.read_csv('./data/freqI_test_multi_comp_comb.csv')

PandasTools.AddMoleculeColumnToFrame(df,'solv_comb_sm', 'solv')
PandasTools.AddMoleculeColumnToFrame(df,'salt_sm', 'salt')

ECFP4_fps_salt = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in df['salt']]
ECFP4_fps_solv = [AllChem.GetMorganFingerprintAsBitVect(x,2) for x in df['solv']]

ECFP4_fps_comb = []

for i in range(len(ECFP4_fps_salt)):
    ECFP4_fps_comb.append(ECFP4_fps_solv[i] + ECFP4_fps_salt[i])

tsne = TSNE(random_state=0).fit_transform(ECFP4_fps_comb)

svgs_salt = [moltosvg(m).data for m in df.salt]

svgs_solv = [moltosvg(m).data for m in df.solv]


ChangeMoleculeRendering(renderer='PNG')


source = ColumnDataSource(data=dict(x=tsne[:,0], y=tsne[:,1], svgs_salt=svgs_salt, svgs_solv=svgs_solv, desc= df.conductivity_log))

hover = HoverTool(tooltips="""
    <div>
        <div> 
            <span style="font-size: 17px; font-weight: bold;"> Salt @svgs_salt{safe} </span>
        </div>
        <div> 
            <span style="font-size: 17px; font-weight: bold;"> Solvent @svgs_solv{safe} </span>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;"> Conductivity @desc </span>
        </div>
    </div>
    """
)
interactive_map = figure(width=1000, height=1000, tools=['reset,box_zoom,wheel_zoom,zoom_in,zoom_out,pan',hover], title="Liquid Electrolytes (ECFP4)")

interactive_map.circle('x', 'y', size=5, source=source, fill_alpha=0.2);

output_file("interactive_map.html")
save(interactive_map)