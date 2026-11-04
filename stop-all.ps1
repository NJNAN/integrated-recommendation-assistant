<#
    拾物 · 多模态推荐助手 全停脚本
    停掉网页服务（8080）、模块一服务（5001）、模块二 worker（8082/8083），
    以及 start.bat -Detach 打开的后台窗口；最后复核端口是否真的释放。
#>
$ErrorActionPreference = 'Continue'
$root = $PSScriptRoot
Set-Location $root
$ports = 8080, 5001, 8082, 8083
$stopped = @()
$names = 'app.py', 'module_one_server.py', 'module_two_worker.py'

Write-Host ''
Write-Host '[全停] 拾物 · 多模态推荐助手'
Write-Host ''

# ① 按端口停（只动 python 进程，避免误杀别的程序）
foreach ($port in $ports) {
    Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | ForEach-Object {
        $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        if ($proc -and $proc.ProcessName -like 'python*') {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Host "[停止] 端口 $port  PID $($proc.Id)"
            $script:stopped += $proc.Id
        }
    }
}

# ② 命令行兜底：抓没有监听端口的进程（例如模块二 IPC worker）
Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    $cmdLine = $_.CommandLine
    if (-not $cmdLine) { return }
    if ($script:stopped -contains $_.ProcessId) { return }
    foreach ($name in $names) {
        if ($cmdLine -like "*$name*" -and $cmdLine -like "*$root*") {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
            Write-Host "[停止] 残留进程 PID $($_.ProcessId)  ($name)"
            $script:stopped += $_.ProcessId
            break
        }
    }
}

# ③ 关掉 start.bat 打开的最小化窗口
Get-Process -Name cmd -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like '拾物-推荐助手*' } | ForEach-Object {
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    Write-Host "[停止] 后台窗口 PID $($_.Id)"
}

# ④ 清理运行期临时文件
Remove-Item (Join-Path $root 'runtime\.mode.tmp') -Force -ErrorAction SilentlyContinue

Start-Sleep -Milliseconds 600

# ⑤ 复核
$left = @()
foreach ($port in $ports) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) { $left += $port }
}
Write-Host ''
if ($script:stopped.Count -eq 0 -and $left.Count -eq 0) {
    Write-Host '没有发现运行中的服务，环境已经是干净的。'
} else {
    Write-Host "已停止 $($script:stopped.Count) 个进程。"
}
if ($left.Count -gt 0) {
    Write-Host "[注意] 这些端口仍被其它程序占用: $($left -join ' ')"
    Write-Host '        不是本项目的进程，脚本不会强杀。需要时手动处理。'
} else {
    Write-Host "端口 $($ports -join ' / ') 均已释放。"
}
Write-Host ''