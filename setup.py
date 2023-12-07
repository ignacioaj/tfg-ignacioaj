from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Trabajo Fin de Grado'
LONG_DESCRIPTION = 'Trabajo Fin de Grado centrado en la detección de cromosomas dicéntricos'

# Configurando
setup(
    # el nombre debe coincidir con el nombre de la carpeta
    # 'modulomuysimple'
    name="tfg-ignacioaj",
    version=VERSION,
    author="Ignacio Atencia-Jiménez",
    author_email="<ignacioatencia@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],  # añade cualquier paquete adicional que debe ser
    # instalado junto con tu paquete. Ej: 'caer'

    keywords=['python', 'computer vision','deep learning','dicentric chromosomes','biological dosimetry','radioactivity'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)