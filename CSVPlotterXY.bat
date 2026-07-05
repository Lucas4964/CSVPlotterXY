@echo off
rem Inicia o CSVPlotterXY a partir do codigo-fonte (sem janela de console).
rem Um caminho de CSV pode ser passado como argumento.
start "" "%~dp0.venv\Scripts\pythonw.exe" -m plotxy_app %*
