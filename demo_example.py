# import RDKit ----------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
drawOptions = Draw.rdMolDraw2D.MolDrawOptions()
drawOptions.prepareMolsBeforeDrawing = False
from rdkit.Chem.Draw import IPythonConsole
import pdb

# define the smiles string and covert it into a molecule sturcture ------------
smiles = '[Li+].C(F)(F)(F)S(=O)(=O)[N-]S(=O)(=O)C(F)(F)F'

mol = Chem.MolFromSmiles(smiles)

# draw the modecule -----------------------------------------------------------
Draw.MolToFile(mol, 'liq_2.png')

# draw the molecule with property ---------------------------------------------
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))

#draw Morgan Fingerprint

bi = {}
fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bi)

tpls = [(mol, x, bi) for x in fp.GetOnBits()]

p = Draw.DrawMorganBits(tpls, molsPerRow=5, legends=[str(x) for x in fp.GetOnBits()], drawOptions=drawOptions)
p.save('liq_2_morgan.png')
    
# Draw.MolToFile(mol, 'liq_2_with_prop.png')
