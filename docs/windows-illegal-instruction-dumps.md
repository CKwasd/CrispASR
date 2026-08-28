# Capturing a Windows illegal-instruction address

Use this when CrispASR exits with `-1073741795` (`0xC000001D`). The exit code
only says that Windows raised `STATUS_ILLEGAL_INSTRUCTION`; the dump should identify
the exact instruction and its offset within `crispasr.exe` or a DLL.
First run the fixed binary normally. If it starts, repeat once from a
Command Prompt with the old eager behavior restored:

```bat
set GGML_CPU_EAGER_UE4M3_LUT=1
rem launch the same crispasr.exe command here
```

Default-starts/eager-crashes is a controlled confirmation that the UE4M3
initialization was the trigger. Clear the variable before testing the default
path again — it persists for the life of that Command Prompt window, so a
"default" run in the same window is still an eager run:

```bat
set GGML_CPU_EAGER_UE4M3_LUT=
```

If the default still crashes, capture its dump as below; a different optimized
CPU path is executing during startup.

## Capture a minimal dump

1. Download Microsoft's Sysinternals ProcDump and create `C:\crispasr-dumps`.

2. Open Command Prompt and run this *before* launching CrispASR. **Give the
   final argument as a dump FILE, not a directory** — pointed at a directory,
   ProcDump does not reliably derive a name and you end up with no dump:

   ```bat
   procdump64.exe -accepteula -e 1 -w crispasr.exe C:\crispasr-dumps\crispasr_dump
   ```

   That writes `C:\crispasr-dumps\crispasr_dump.dmp` — ProcDump appends the
   extension itself.

   `-e 1` catches first-chance exceptions as well as unhandled ones. Plain `-e`
   only fires on an *unhandled* exception, and parts of the CUDA and ggml stack
   install their own handlers, so a bare `-e` can sit there and produce nothing.
   The tradeoff is that `-e 1` may also dump an exception the process handles and
   recovers from, so confirm the process actually died before sending anything.

   If you are running the 32-bit `procdump.exe` rather than `procdump64.exe`,
   add `-64` to force a 64-bit dump of the 64-bit process.

3. Reproduce the crash once. ProcDump's default mini dump includes the exception
   context, thread stacks, module list, and referenced memory. Do not use `-ma`
   unless requested: a full dump may contain model data, audio, paths, and other
   process memory.

4. Record the exact CrispASR release/filename and the CPU's **instruction-set
   flags**. The flags matter more than the model name, because the usual cause of
   `0xC000001D` is a build compiled for a wider ISA than the CPU supports (#380).

   Sysinternals Coreinfo reports them natively, and you are already downloading
   from Sysinternals for ProcDump:

   ```bat
   Coreinfo64.exe -f -accepteula
   ```

   The model name is still worth including:

   ```powershell
   Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
   ```

   From MSYS/Git-Bash, if you prefer:

   ```bash
   grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512[a-z_]*|avx_vnni|fma|f16c|sse4_[12]|ssse3|bmi[12])$' | sort
   ```

   No `avx2` in that list means the AVX2 build cannot run: use the
   `crispasr-windows-x86_64-cpu-legacy` package. From v0.8.30 the CLI detects
   this itself and says so instead of dying silently.

## If step 2 produced no dump file

Windows Error Reporting can capture the process regardless of how it exits, with
nothing to download. Create this key and reproduce the crash again:

```
HKLM\SOFTWARE\Microsoft\Windows\Windows Error Reporting\LocalDumps\crispasr.exe
  DumpFolder  (REG_EXPAND_SZ)  C:\crispasr-dumps
  DumpType    (REG_DWORD)      1     ; 1 = mini, 2 = full
```

Creating the key requires an elevated Command Prompt or Registry Editor. Leave
`DumpType` at `1`; `2` carries the same privacy concerns as ProcDump's `-ma`.
Delete the key once you have the dump.

## Extract the useful address locally

Open the `.dmp` in WinDbg. If `windbg` is not on your `PATH`, note that there are
two builds: the classic `windbg.exe` from Debugging Tools for Windows in the
Windows SDK, and the modern Store app, which installs as **`WinDbgX.exe`** under
`%LOCALAPPDATA%\Microsoft\WindowsApps`. Either works.

```text
.logopen /t crispasr-illegal-instruction.txt
.symfix+
.sympath+ C:\real\path\to\crispasr\build
.exepath+ C:\real\path\to\crispasr\build
.reload /f crispasr.exe
.reload /f ggml-cpu.dll
!analyze -v
.exr -1
.ecxr
r
u @rip L10
db @rip L20
k
~*k
lmvm crispasr
lmvm ggml*
.logclose
```

`!analyze -v` cannot resolve an instruction inside `ggml-cpu.dll` until the
debugger knows where the binaries are, and left to itself it will try to
download unrelated symbols instead. The `.exepath`/`.sympath`/`.reload`
lines above point it at the directory you unzipped — give the folder holding
`crispasr.exe` and `ggml-cpu.dll`, not a single DLL. (`.load` is for debugger
*extensions*; it will not attach a module's symbols.)

Note that `.reload /f` is scoped to the two modules that matter. A bare
`.reload /f` force-loads everything, and the NVIDIA modules (`nvcuda.dll`,
`nvapi64.dll`, `nvdxgdmal64.dll`, `cudart64_12.dll`, `cublas64_12.dll`, and so
on) will never resolve on any machine but the one that produced the dump. Those
errors are expected noise; scoping the reload keeps them out of the log so the
failures that remain are the ones worth reading.

`.symfix+` adds the Microsoft symbol server so ntdll and kernel32 frames
resolve. Drop that line on an offline machine.

## What a useful result looks like

The instruction at `RIP` should be a wide vector op — something like
`vpdpbusd`, `vfmadd231ps`, or any `v*` instruction with a `zmm` or `ymm`
operand — and the module containing `RIP` should be `ggml-cpu.dll`. That is the
ISA-mismatch signature.

`db @rip L20` is there because the raw bytes still identify the instruction when
the disassembler chokes on an encoding it does not know.

If `RIP` lands in `nvcuda.dll` or another driver module instead, this is not an
ISA mismatch and the triage goes elsewhere: report the driver version alongside
the log.

## Before you post

Scrub the log first. `lmvm`, `.exepath`, `~*k`, and `!analyze -v` all emit local
paths, and `!analyze -v` includes the process command line — so the file will
usually contain your Windows username, and may contain model or audio paths.
Skim it and redact before attaching.

Then send `crispasr-illegal-instruction*.txt`, the CPU name and flags, and the
exact binary — not the dump — in the public issue. The key fields are the
exception address, the module containing `RIP`, the disassembled instruction at
`RIP`, and the module load address. Those let maintainers calculate an
ASLR-independent module offset and resolve it against the matching
executable/PDB.

If the text is insufficient, share the `.dmp` privately. Even a mini dump can
contain local paths or small referenced buffers, so do not attach it publicly
without reviewing that risk.
