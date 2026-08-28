# Capturing a Windows illegal-instruction address

Use this when CrispASR exits with `-1073741795` (`0xC000001D`). The exit code
only says that Windows raised `STATUS_ILLEGAL_INSTRUCTION`; the dump identifies
the exact instruction and its offset within `crispasr.exe` or a DLL.

For #302, first run the fixed binary normally. If it starts, repeat once from a
Command Prompt with the old eager behavior restored:

```bat
set GGML_CPU_EAGER_UE4M3_LUT=1
rem launch the same crispasr.exe command here
```

Default-starts/eager-crashes is a controlled confirmation that the UE4M3
initialization was the trigger. If the default still crashes, capture its dump
as below; a different optimized CPU path is executing during startup.

## Capture a minimal dump

1. Download Microsoft's Sysinternals ProcDump and create `C:\crispasr-dumps`.
2. Open Command Prompt and run this *before* launching CrispASR. **Give the
   final argument as a dump FILE, not a directory** — pointed at a directory,
   ProcDump does not reliably derive a name and you end up with no dump
   (reported in #403):

   ```bat
   procdump64.exe -accepteula -64 -e -w crispasr.exe C:\crispasr-dumps\crispasr_dump
   ```

   That writes `C:\crispasr-dumps\crispasr_dump.dmp` — ProcDump appends the
   extension itself. `-64` forces a 64-bit dump of the 64-bit process, which
   matters when a 32-bit ProcDump is first on `PATH`.

3. Reproduce the crash once. ProcDump's default mini dump includes the exception
   context, thread stacks, module list, and referenced memory. Do not use `-ma`
   unless requested: a full dump may contain model data, audio, paths, and other
   process memory.
4. Record the exact CrispASR release/filename and the CPU name:

   ```powershell
   Get-CimInstance Win32_Processor | Select-Object -ExpandProperty Name
   ```

   The **instruction-set flags matter more than the model name**, because the
   usual cause of `0xC000001D` is a build compiled for a wider ISA than the CPU
   supports (#380). From MSYS/Git-Bash:

   ```bash
   grep -m1 '^flags' /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512[a-z]*|fma|f16c|sse4_[12]|ssse3|bmi[12])$' | sort
   ```

   No `avx2` in that list means the AVX2 build cannot run: use the
   `crispasr-windows-x86_64-cpu-legacy` package. From v0.8.30 the CLI detects
   this itself and says so instead of dying silently.

## Extract the useful address locally

Open the `.dmp` in WinDbg — on Windows 11 the executable is **`WinDbgX.exe`**
(the older `windbg.exe` name is not what ships) — and run:

```text
.logopen /t crispasr-illegal-instruction.txt
.exepath+ C:\path\to\your\crispasr
.sympath+ C:\path\to\your\crispasr
.reload /f
!analyze -v
.exr -1
.ecxr
r
ub @rip L8
u @rip L8
k
lm
.logclose
```

`!analyze -v` cannot resolve an instruction inside `ggml-cpu.dll` until the
debugger knows where the binaries are, and left to itself it will try to
download unrelated symbols instead (#403). The `.exepath`/`.sympath`/`.reload`
lines above point it at the directory you unzipped — give the folder holding
`crispasr.exe` and `ggml-cpu.dll`, not a single DLL. (`.load` is for debugger
*extensions*; it will not attach a module's symbols.)

Send `crispasr-illegal-instruction*.txt`, the CPU name, and the exact binary—not
the dump—in the public issue. The key fields are the exception address, the
module containing `RIP`, the disassembled instruction at `RIP`, and the module
load address. Those let maintainers calculate an ASLR-independent module offset
and resolve it against the matching executable/PDB.

If the text is insufficient, share the `.dmp` privately. Even a mini dump can
contain local paths or small referenced buffers, so do not attach it publicly
without reviewing that risk.
