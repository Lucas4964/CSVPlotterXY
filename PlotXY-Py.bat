@echo off
rem Inicia o PlotXY-Py (sem janela de console). Um caminho de CSV pode ser passado como argumento.
start "" "%~dp0.venv\Scripts\pythonw.exe" -m plotxy_app %*
