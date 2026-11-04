@echo off
chcp 936 >nul
rem ============================================================
rem  拾物 · 多模态推荐助手 一键启动（Windows 批处理）
rem
rem  start.bat              在线模式：自动接模块一，真实 2208 件商品
rem  start.bat demo         离线模式：18 件合成商品，不启模块一
rem  start.bat detach       后台启动，脚本立即返回（日志 runtime\server.out.log）
rem  start.bat stop         停止服务（网页端口 + 模块一 5001 一起清）
rem  start.bat 8090         换端口启动
rem  start.bat demo detach  组合使用
rem ============================================================
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

set "MODE=auto"
set "ACTION=run"
set "PORT=8080"

:parse
if "%~1"=="" goto prepare
if /i "%~1"=="demo"   ( set "MODE=demo"   & shift & goto parse )
if /i "%~1"=="auto"   ( set "MODE=auto"   & shift & goto parse )
if /i "%~1"=="detach" ( set "ACTION=detach" & shift & goto parse )
if /i "%~1"=="stop"   ( set "ACTION=stop"   & shift & goto parse )
echo %~1|findstr /r "^[0-9][0-9]*$" >nul
if not errorlevel 1 ( set "PORT=%~1" & shift & goto parse )
echo 未知参数: %~1
echo 用法: start.bat [auto^|demo] [detach^|stop] [端口]
exit /b 2

:prepare
echo.

if /i "%ACTION%"=="stop" goto stop_service

where python >nul 2>&1
if errorlevel 1 (
  echo [错误] 未找到 python，请安装 Python 3.10+ 并加入 PATH。
  exit /b 1
)

netstat -ano -p tcp | findstr /r /c:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo [错误] 端口 %PORT% 已被占用。换端口: start.bat %PORT%   或先停止: start.bat stop
  exit /b 1
)

if /i not "%MODE%"=="demo" call :clean_stale_module_one

set "IRA_PORT=%PORT%"
set "IRA_PROFILE="
if /i "%MODE%"=="demo" (
  set "IRA_PROFILE=demo"
  echo [模式] 离线演示：18 件合成商品，不启动模块一
) else (
  if exist "vendor\01_recommender_system\release\web_system\data\catalog.csv" (
    echo [模式] 自动接入模块一：2208 件真实商品 + 冻结排序（就绪需 5~20 秒）
  ) else (
    echo [警告] 没找到模块一发布包，将自动回退离线演示
  )
)

if not exist runtime mkdir runtime

if /i "%ACTION%"=="detach" (
  start "拾物-推荐助手" /min cmd /c "set IRA_PORT=%PORT%&&set IRA_PROFILE=%IRA_PROFILE%&&python -B app.py > runtime\server.out.log 2>&1"
  echo [启动] 后台运行中，日志: runtime\server.out.log
  call :wait_health
  if errorlevel 1 (
    echo [错误] 60 秒内未就绪，看日志: runtime\server.out.log
    exit /b 1
  )
  set "HMODE="
  powershell -NoProfile -Command "try{(Invoke-RestMethod 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2).mode}catch{''}" > "runtime\.mode.tmp" 2>nul
  set /p HMODE=<"runtime\.mode.tmp"
  del "runtime\.mode.tmp" >nul 2>&1
  if /i "%MODE%"=="auto" if /i "!HMODE!"=="offline_demo" (
    echo.
    echo [警告] 模块一没接上，当前是离线演示模式。
    echo        多半是 5001 端口被别的进程占着：先 start.bat stop 再启动。
  )
  echo.
  echo   地址 : http://127.0.0.1:%PORT%
  echo   模式 : %MODE%  ^(实际: !HMODE!^)
  echo   停止 : start.bat stop
  echo.
  start "" http://127.0.0.1:%PORT%
) else (
  echo [启动] http://127.0.0.1:%PORT%    ^(Ctrl+C 结束服务^)
  echo.
  python -B app.py
  exit /b %errorlevel%
)
exit /b 0

:clean_stale_module_one
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 5001 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { $p = Get-CimInstance Win32_Process -Filter ('ProcessId=' + $_.OwningProcess); if ($p.CommandLine -like '*module_one_server.py*') { Stop-Process -Id $p.ProcessId -Force; Write-Host ('[清理] 上次残留的模块一进程 PID ' + $p.ProcessId) } }" 2>nul
exit /b 0

:stop_service
set "FOUND=0"
for %%P in (%PORT% 5001) do (
  for /f "tokens=5" %%I in ('netstat -ano -p tcp ^| findstr /r /c:":%%P .*LISTENING"') do (
    taskkill /f /pid %%I >nul 2>&1
    if not errorlevel 1 (
      echo [停止] 端口 %%P  PID %%I
      set "FOUND=1"
    )
  )
)
if "!FOUND!"=="0" ( echo 没有发现本项目的运行中服务。 ) else ( echo 已停止。 )
exit /b 0

:wait_health
set /a TRY=0
:wait_loop
powershell -NoProfile -Command "try{$null=Invoke-RestMethod 'http://127.0.0.1:%PORT%/api/health' -TimeoutSec 2; exit 0}catch{exit 1}" >nul 2>&1
if not errorlevel 1 exit /b 0
set /a TRY+=1
if !TRY! GEQ 60 exit /b 1
ping -n 2 127.0.0.1 >nul
goto wait_loop
