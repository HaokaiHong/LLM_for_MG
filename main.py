import streamlit as st 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole

import requests
import json
import time

# SET you key
key = ''


def ask_gpt(prompt_content, key):
    url = "https://openai.api2d.net/v1/chat/completions"

    headers = {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer '+key # <-- 把 fkxxxxx 替换成你自己的 Forward Key，注意前面的 Bearer 要保留，并且和 Key 中间有一个空格。
    }

    data = {
      "model": "gpt-3.5-turbo",
      "messages": [{"role": "user", "content": prompt_content}]
    }
    response = requests.post(url, headers=headers, json=data)
    print(response)
    answer = response.json()['choices'][0]['message']['content']
    print(answer)
    return answer


def draw_molecule_2d(smiles):
    st.write(f"- For SMILES: {smiles} \n - The chemical structure of the molecule is as follows:")
    mol = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(mol)
    Chem.Draw.MolToFile(mol, "molecule_2d.png")
    image = plt.imread("molecule_2d.png")
    st.image(image)


def predict_property(smiles, key):
    iupac_formula = 'Given the SMILES "' + smiles + '" of a compound, generate its IUPAC name, chemical formula, Molecular mass, common name, CAS number, melting point, boiling point, solubility, and chemical class. Then predict its Polarizability,highest occupied molecular orbital energy, lowest unoccupied molecular orbital energy, and dipole moment. Return the answer in the Table with column header'
    iupac_formula_response = ask_gpt(iupac_formula, key)
    st.write(iupac_formula_response)


def design_new_molecule(smiles, key):
    generation_prompt = 'Given the SMILES "' + smiles + '"  of a compound, design a similar but novel molecule base on this molecule, return designed new SMILES in the format of "SMILES: \nDesign reason:"'
    generation = ask_gpt(generation_prompt, key)
    for text in generation.split('\n'):
        if 'SMILES' in text:
            generated_smiles = text[8:]
        if 'Design reason' in text:
            design_reason = text[15:]
    st.write(f"- Generate a new molecule: {generated_smiles}\n")
    st.write(f"- Design reason: {design_reason}")
    return generated_smiles


st.title('Molecule Explainer and Generator by LLM')

st.sidebar.write('## Input a SMILES')

smile_text = st.sidebar.text_input('SMILES for Explanation and Generation', '')
st.sidebar.write('### Here are some examples \n - C(O)C \n - OCC \n - CC(=O)OC1=CC=CC=C1C(=O)O \n - OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N')
if smile_text != '':

    st.write(f"## 1. Visualization")


    draw_molecule_2d(smile_text)

    st.write('## 2. Property Prediction')
    predict_property(smile_text, key)


    st.write('## 3. Molecule Generatioin')
    generated_smiles = design_new_molecule(smile_text, key)

    st.write('## 4. Visualization and Property Prediction for Designed Molecule')
    draw_molecule_2d(generated_smiles)
    predict_property(generated_smiles, key)

