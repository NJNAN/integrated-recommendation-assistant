@echo off
chcp 936 >nul
rem ============================================================
rem  拾物 · 多模态推荐助手 全停脚本
rem
rem  stop-all.bat   停掉网页服务、模块一、模块二 worker 和后台窗口
rem ============================================================
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop-all.ps1"
exit /b %errorlevel%