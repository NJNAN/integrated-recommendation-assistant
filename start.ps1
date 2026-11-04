<#
.SYNOPSIS
    一键启动「拾物 · 多模态推荐助手」（模块三整合项目）。

.DESCRIPTION
    auto 模式（默认）：自动拉起模块一冻结服务（127.0.0.1:5001），使用 2,208 件真实商品
                       与 Oblivionis v2 冻结排序；模块一不可用时自动回退离线，不会启动失败。
    demo 模式：纯离线，只用 data/catalog.json 的 18 件合成商品，不启模块一、不依赖 Flask。

.PARAMETER Port
    网页服务端口，默认 8080。

.PARAMETER Detach
    后台启动：脚本退出后服务继续运行，日志写到 runtime\server.*.log，并打印停止命令。

.PARAMETER NoOpen
    不自动打开浏览器。

.PARAMETER Stop
    停止正在运行的服务（网页端口 + 模块一 5001 子进程都清掉）。

.EXAMPLE
    .\start.ps1
    真实商品模式启动，浏览器打开 http://127.0.0.1:8080

.EXAMPLE
    .\start.ps1 -Mode demo
    离线演示模式（最快，答辩断网也能跑）

.EXAMPLE
    .\start.ps1 -Detach -Port 8090
    后台启动在 8090，脚本立即返回
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'demo')][string]$Mode = 'auto',
    [int]$Port = 8080,
    [switch]$Detach,
    [switch]$NoOpen,
    [switch]$Stop
)

$ErrorActionPreference = 'Stop'
$root = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
Set-Location $root
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

# ⓪ -Stop：清掉网页端口和模块一子进程（强杀主进程时子进程会变孤儿，所以两个都查）
if ($Stop) {
    $stopped = 0
    foreach ($targetPort in @($Port, 5001)) {
        $conns = Get-NetTCPConnection -LocalPort $targetPort -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            $procInfo = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
            if ($procInfo -and $procInfo.ProcessName -like 'python*') {
                Stop-Process -Id $procInfo.Id -Force
                Write-Host "[停止] 端口 $targetPort → PID $($procInfo.Id)"
                $stopped++
            }
        }
    }
    if ($stopped -eq 0) { Write-Host '没有发现本项目的运行中服务。' } else { Write-Host "已停止 $stopped 个进程。" }
    return
}

# ① 找到 Python（3.10+；离线核心只用标准库）
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    throw '未找到 python。请安装 Python 3.10+ 并把 python.exe 加入 PATH。'
}

# ② 端口占用要提前报错，而不是让服务静默起不来
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    $busyPids = ($busy | Select-Object -ExpandProperty OwningProcess -Unique) -join ', '
    throw "端口 $Port 已被占用（PID: $busyPids）。换一个：.\start.ps1 -Port 8090"
}

# ③ 运行模式
# 先清理上次非正常退出留下的模块一进程：5001 被占会让本次自动降级成离线
if ($Mode -ne 'demo') {
    Get-NetTCPConnection -LocalPort 5001 -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        $stale = Get-CimInstance Win32_Process -Filter "ProcessId=$($_.OwningProcess)" -ErrorAction SilentlyContinue
        if ($stale -and $stale.CommandLine -like '*module_one_server.py*') {
            Stop-Process -Id $stale.ProcessId -Force
            Write-Host "[清理] 上次残留的模块一进程 PID $($stale.ProcessId)" -ForegroundColor DarkGray
        }
    }
}
$env:IRA_PORT = "$Port"
# 统一用 UTF-8 输出：Windows 下 python 重定向到文件默认用 GBK，中文日志会乱码
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$moduleOneCatalog = Join-Path $root 'vendor/01_recommender_system/release/web_system/data/catalog.csv'
if ($Mode -eq 'demo') {
    $env:IRA_PROFILE = 'demo'
    Write-Host '[模式] 离线演示：18 件合成商品，不启动模块一' -ForegroundColor Cyan
} else {
    Remove-Item Env:IRA_PROFILE -ErrorAction SilentlyContinue
    if (Test-Path $moduleOneCatalog) {
        Write-Host '[模式] 自动接入模块一：2,208 件真实商品 + 冻结排序（就绪需 5~20 秒）' -ForegroundColor Cyan
    } else {
        Write-Host '[警告] 没找到模块一发布包，将自动回退离线演示' -ForegroundColor Yellow
    }
}

$url = "http://127.0.0.1:$Port"

# ④ 启动
if ($Detach) {
    $logDir = Join-Path $root 'runtime'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stdoutLog = Join-Path $logDir 'server.out.log'
    $stderrLog = Join-Path $logDir 'server.err.log'
    # 用 cmd 做重定向：Start-Process 自带的 -RedirectStandardOutput 会让父 shell 一直等子进程退出，
    # 导致脚本打印完不返回。这里让 cmd 负责重定向，父 shell 立刻拿回控制权。
    $cmdLine = '/c ""{0}" -B app.py > "{1}" 2>&1"' -f $python.Source, $stdoutLog
    $shell = Start-Process -FilePath 'cmd.exe' -ArgumentList $cmdLine `
        -WorkingDirectory $root -WindowStyle Hidden -PassThru
    Start-Sleep -Milliseconds 800
    $child = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($shell.Id)" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'python*' } | Select-Object -First 1
    $serverPid = if ($child) { $child.ProcessId } else { $shell.Id }
    Write-Host "[启动] 后台进程 PID=$serverPid   日志: runtime\server.out.log" -ForegroundColor DarkGray

    # 轮询 /api/health 直到就绪（最多 60 秒）
    $deadline = (Get-Date).AddSeconds(60)
    $health = $null
    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $serverPid -ErrorAction SilentlyContinue)) {
            $tail = if (Test-Path $stdoutLog) { (Get-Content $stdoutLog -Encoding UTF8 -Tail 10) -join [Environment]::NewLine } else { '' }
            throw "服务进程已退出。日志尾部：`n$tail"
        }
        try { $health = Invoke-RestMethod "$url/api/health" -TimeoutSec 2; break }
        catch { Start-Sleep -Milliseconds 500 }
    }
    if (-not $health) { throw "等了 60 秒仍未就绪。看日志：$stdoutLog" }

    $modeText = if ($health.mode -eq 'hybrid_configured') { '真实商品（模块一已接入）' } else { '离线演示' }
    Write-Host ''
    Write-Host "  地址   : $url" -ForegroundColor Green
    Write-Host "  模式   : $modeText"
    Write-Host "  数据集 : $($health.dataset)（$($health.catalog_size) 件商品，$($health.currency)）"
    Write-Host "  停止   : Stop-Process -Id $serverPid" -ForegroundColor DarkGray
    Write-Host ''
} else {
    Write-Host "[启动] $url    （Ctrl+C 结束服务）" -ForegroundColor Green
    & $python -B app.py
    exit $LASTEXITCODE
}

if (-not $NoOpen) { Start-Process $url }
