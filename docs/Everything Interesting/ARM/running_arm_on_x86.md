> 因为《嵌入式系统》需要写一段简单的 ARM 程序作为课程设计，提供的 ARM Developer Suite 又十分古老难用，所以找到了这样一招。
>
> 摘抄的这篇文章讲述了如何使用 `QEMU`  运行在 `x86`  机器上运行 `ARM`  的二进制程序。

# RUNNING ARM BINARIES ON X86 WITH QEMU-USER

Ever wanted to play around with Arm assembly without an Arm board and the hassle of setting up a full-system QEMU emulation?

This blog post is a quick and straight-forward way to compile, debug, and run Arm 32- and 64-bit binaries directly on your x86_64 Linux host system. Full system emulation has its benefits, especially if you want a dedicated environment to tinker around with things like firmware emulation. If you are looking for a quicker way to play around with Arm assembly, run your Arm binaries, and perform simple debugging tasks, the QEMU user mode emulation is more than sufficient.

FYI: If you are looking for a full-system emulation and want to save time, you can download my [Lab VM 2.0](https://azeria-labs.com/lab-vm-2-0/) which contains an Armv7-A emulation. But keep in mind that this is an emulation of a 32-bit architecture.

EXECUTING ARM64 BINARIES (C TO BINARY)

FYI: In this tutorial, I’m using an [Ubuntu 20.04.1 LTS](https://ubuntu.com/download/desktop) VM as a host system.

Since processors don’t understand high-level source code directly, we need to convert our C code into machine-code using a compiler. However the GCC compiler you have on your system compiles your code for the architecture of the system it runs on, in this case x86_64. In order to compile our code for the Arm architecture, we need to use a cross-compiler.

Let’s start with Arm64 and install the following packages:

```
azeria@ubuntu:~$ sudo apt update -y && sudo apt upgrade -y
azeria@ubuntu:~$ sudo apt install qemu-user qemu-user-static gcc-aarch64-linux-gnu binutils-aarch64-linux-gnu binutils-aarch64-linux-gnu-dbg build-essential
```

Once installed, create a file containing a simple C program for testing, e.g.

```
#include <stdio.h>

int main(void) {
    return printf("Hello, I'm executing ARM64 instructions!\n");
}
```

To compile the code as a static executable, we can use _aarch64-linux-gnu-gcc_ with the -static flag.

```
azeria@ubuntu:~$ aarch64-linux-gnu-gcc -static -o hello64 hello.c
```

But what happens if we run this Arm executable on a different architecture? Executing it on an x86_64 architecture would normally result in an error telling us that the binary file cannot be executed due to an error in the executable format.

```
azeria@ubuntu:~$ ./hello64
bash: ./hello64: cannot execute binary file: Exec format error
```

We can’t run our Arm binary on an x84_64 architecture because instructions are encoded differently on these two architectures.

Lucky for us, we can bypass this restriction with the QEMU user emulator which allows us to run binaries for other architectures on our host system. Let’s try it out.

Below you can see that our host is a x86_64 GNU/Linux system. The binary we have previously compiled is ARM aarch64.

```
azeria@ubuntu:~$ uname -a
Linux ubuntu 5.4.0-58-generic #64-Ubuntu SMP Mon Dec 29 08:16:25 UTC 2020 x86_64 x86_64 x86_64 GNU/Linux
azeria@ubuntu:~$ file hello64
hello64: ELF 64-bit LSB executable, ARM aarch64, version 1 (GNU/Linux), statically linked, BuildID[sha1]=66307a9ec0ecfdcb05002f8ceecd310cc6f6792e, for GNU/Linux 3.7.0, not stripped
```

Let’s execute it!

```
azeria@ubuntu:~$ ./hello64
Hello, I'm executing ARM64 instructions!
```

Voilà, our statically linked aarch64 binary is running on our x86*64 host thanks to \_qemu-user-static*. But can we execute a dynamically linked Arm executable? Yes, we can. This time, the package that makes this possible is _qemu-user_.

First, compile the C code without the _-static_ flag. In order to run it, we need to use _qemu-aarch64_ and supply the aarch64 libraries via the _-L_ flag.

```
azeria@ubuntu:~$ aarch64-linux-gnu-gcc -o hello64dyn hello64.c
azeria@ubuntu:~$ qemu-aarch64 -L /usr/aarch64-linux-gnu ./hello64dyn
Hello, I'm executing ARM64 instructions!
```

Nice. Works like a charm. Moving on to Arm32!

EXECUTING ARM32 BINARIES (C TO BINARY)

The same procedure applies to Arm 32-bit binaries, but we need to install different packages (in addition to the previously installed _qemu-user_ packages).

```
$ sudo apt install gcc-arm-linux-gnueabihf binutils-arm-linux-gnueabihf binutils-arm-linux-gnueabihf-dbg
```

We’ll use the same simple C program as before and call it hello32.c:

```
#include <stdio.h>

int main(void) {
    return printf("Hello, I am an ARM32 binary!\n");
}
```

Now we compile this program as a statically linked Arm32 executable using _arm-linux-gnueabihf-gcc_ with the _-static flag_ and run it:

```
azeria@ubuntu:~$ arm-linux-gnueabihf-gcc -static -o hello32 hello32.c
azeria@ubuntu:~$ ./hello32
Hello, I am an ARM32 binary!
```

Now let’s compile it as a dynamically linked executable.

```
azeria@ubuntu:~$ arm-linux-gnueabihf-gcc -o hello32 hello32.c
azeria@ubuntu:~$ qemu-arm -L /usr/arm-linux-gnueabihf ./hello32
Hello, I am an ARM32 binary!
```

EXECUTING ARM BINARIES (ASSEMBLY TO BINARY)

Now that we know how to compile code for the Arm architecture and run it on an x86_64 host, let’s try this with assembly source code.

Since processors only understand machine code and not assembly language directly, we need a program to convert our hand-written assembly instructions into their machine-code equivalents. The programs that perform this task are called _assemblers_.

There are different assemblers available on different platforms, such as the [GNU assembler “as”](https://sourceware.org/binutils/docs/as/GNU-Assembler.html#GNU-Assembler) which is also used to assemble the Linux kernel, the ARM Toolchain assembler “[armasm](https://developer.arm.com/documentation/dui0473/j/using-the-assembler?lang=en)”, or the Microsoft assembler with the same name (“[armasm](https://docs.microsoft.com/en-us/cpp/assembler/arm/arm-assembler-reference?view=msvc-160)”) included in Visual Studio. Here we will use the GNU assembler.

Suppose we want to assemble the following hello world assembly program:

```
.section .text
.global _start

_start:
/* syscall write(int fd, const void *buf, size_t count) */
    mov x0, #1
    ldr x1, =msg
    ldr x2, =len
    mov w8, #64
    svc #0

/* syscall exit(int status) */
    mov x0, #0
    mov w8, #93
    svc #0

msg:
.ascii "Hello, ARM64!\n"
len = . - msg
```

Normally we would assemble and link it with the native _AS_ and _LD_. But the native assembler can only interpret instructions of the architecture it was build to interpret, e.g. x86_64. Trying to assemble Arm instructions would result in errors:

```
azeria@ubuntu:~$ as asm64.s -o asm64.o && ld asm64.o -o asm64-2
asm64.s: Assembler messages:
asm64.s:6: Error: expecting operand after ','; got nothing
asm64.s:7: Error: no such instruction: `ldr x1,=msg'
asm64.s:8: Error: no such instruction: `ldr x2,=len'
asm64.s:9: Error: expecting operand after ','; got nothing
asm64.s:10: Error: no such instruction: `svc '
asm64.s:13: Error: expecting operand after ','; got nothing
asm64.s:14: Error: expecting operand after ','; got nothing
asm64.s:15: Error: no such instruction: `svc
```

That’s why we need to use a cross-assembler and linker specifically for the instruction set of our program. In this case, A64:

```
azeria@ubuntu:~$ aarch64-linux-gnu-as asm64.s -o asm64.o && aarch64-linux-gnu-ld asm64.o -o asm64
azeria@ubuntu:~$ ./asm64
Hello, ARM64!
```

Let’s do the same for the A32 version of this program:

```
.section .text
.global _start

_start:
/* syscall write(int fd, const void *buf, size_t count) */
    mov r0, #1
    ldr r1, =msg
    ldr r2, =len
    mov r7, #4
    svc #0

/* syscall exit(int status) */
    mov r0, #0
    mov r7, #1
    svc #0

msg:
.ascii "Hello, ARM32!\n"
len = . - msg
```

Assemble and link and…

```
azeria@ubuntu:~$ arm-linux-gnueabihf-as asm32.s -o asm32.o && arm-linux-gnueabihf-ld -static asm32.o -o asm32
azeria@ubuntu:~$ ./asm32
Hello, ARM32!
```

Voilà!

DISASSEMBLE ARM BINARIES ON X86_64

Now that we can compile and run Arm binaries on our host system, let’s take them apart.

The easiest way to look at the disassembly of an ELF binary is with a tool called [_objdump_](https://sourceware.org/binutils/docs/binutils/objdump.html). This is especially useful for small binaries.

But what happens if we use the native _objdump_ from our host system to disassemble an Arm binary?

```
azeria@ubuntu:~$ objdump -d hello32
hello32: file format elf32-little
objdump: can't disassemble for architecture UNKNOWN!
```

Since the native _objdump_ expects a binary compiled for the architecture it is running on (x86*64 in this case), it does not recognize the architecture of the Arm binary we supplied and refuses to disassemble it. But the \_objdump* binary itself does not need to be compiled for the Arm architecture, it only needs to be able to interpret Arm machine code. So all we need is a cross-built of it. If you type “arm-linux” in your terminal and double-tap, you will see all the utilities we have already installed with the _binutils-arm-linux-gnueabihf_ package, and _objdump_ is included!

```
azeria@ubuntu:~$ arm-linux-gnueabihf-
arm-linux-gnueabihf-addr2line      arm-linux-gnueabihf-gcov-9
arm-linux-gnueabihf-ar             arm-linux-gnueabihf-gcov-dump
arm-linux-gnueabihf-as             arm-linux-gnueabihf-gcov-dump-9
arm-linux-gnueabihf-c++filt        arm-linux-gnueabihf-gcov-tool
arm-linux-gnueabihf-cpp            arm-linux-gnueabihf-gcov-tool-9
arm-linux-gnueabihf-cpp-9          arm-linux-gnueabihf-gprof
arm-linux-gnueabihf-dwp            arm-linux-gnueabihf-ld
arm-linux-gnueabihf-elfedit        arm-linux-gnueabihf-ld.bfd
arm-linux-gnueabihf-gcc            arm-linux-gnueabihf-ld.gold
arm-linux-gnueabihf-gcc-9          arm-linux-gnueabihf-nm
arm-linux-gnueabihf-gcc-ar         arm-linux-gnueabihf-objcopy
arm-linux-gnueabihf-gcc-ar-9       arm-linux-gnueabihf-objdump
arm-linux-gnueabihf-gcc-nm         arm-linux-gnueabihf-ranlib
arm-linux-gnueabihf-gcc-nm-9       arm-linux-gnueabihf-readelf
arm-linux-gnueabihf-gcc-ranlib     arm-linux-gnueabihf-size
arm-linux-gnueabihf-gcc-ranlib-9   arm-linux-gnueabihf-strings
arm-linux-gnueabihf-gcov           arm-linux-gnueabihf-strip
```

Now all we have to do is use _arm-linux-gnueabihf-objdump_. Let’s try this with the asm32 binary:

```
azeria@ubuntu:~$ arm-linux-gnueabihf-objdump -d asm32

asm32: file format elf32-littlearm

Disassembly of section .text:

00010054 <_start>:
10054: e3a00001 mov r0, #1
10058: e59f1024 ldr r1, [pc, #36] ; 10084 <msg+0x10>
1005c: e59f2024 ldr r2, [pc, #36] ; 10088 <msg+0x14>
10060: e3a07004 mov r7, #4
10064: ef000000 svc 0x00000000
10068: e3a00000 mov r0, #0
1006c: e3a07001 mov r7, #1
10070: ef000000 svc 0x00000000

00010074 <msg>:
10074: 6c6c6548 .word 0x6c6c6548
10078: 41202c6f .word 0x41202c6f
1007c: 32334d52 .word 0x32334d52
10080: 00000a21 .word 0x00000a21
10084: 00010074 .word 0x00010074
10088: 0000000e .word 0x0000000e
```

…and it works!

DEBUGGING ARM BINARIES ON X86_64

We can also debug these binaries on our host system, but not with the native GDB installation. For our Arm binaries, we will use _gdb-multiarch_.

```
azeria@ubuntu:~$ sudo apt install gdb-multiarch qemu-user
```

We can also compile our C code with the _-ggdb3_ flag which produces additional debugging information for GDB. Let’s compile a statically linked executable for this example:

```
azeria@ubuntu:~$ arm-linux-gnueabihf-gcc -ggdb3 -o hello32-static hello32.c -static
```

One of the ways we can debug this binary is to use the _qemu-user_ emulator and have tell GDB to connect to it through a TCP port. To do this, we run _qemu-arm_ with the _-g_ flag and a port number on which it should wait for a GDB connection. The _-L_ flag sets the ELF interpreter prefix to the path we supply.

```
azeria@ubuntu:~$ qemu-arm -L /usr/arm-linux-gnueabihf -g 1234 ./hello32-static
```

Open another terminal window and use the following command:

```
azeria@ubuntu:~$ gdb-multiarch -q --nh -ex 'set architecture arm' -ex 'file hello32-static' -ex 'target remote localhost:1234' -ex 'layout split' -ex 'layout regs'
```

The _–nh_ flag instructs it to not read the _.gdbinit_ file (it can get buggy if you have a GDB wrapper installed), and the _-ex_ options are the commands we want _gdb-multiarch_ to set at the start of the session. The first one sets the target architecture to arm (use arm64 for 64-bit binaries), then we provide the binary itself, tell it where to find the binary running in our _qemu-arm_ emulation. The final two commands are used to split and display the source, disassembly, command, and register windows.

[![img](media/running_arm_on_x86/gdb-multiarch-1024x763.png.pagespeed.ce.jO1vBh7Koi.png)](https://azeria-labs.com/wp-content/uploads/2020/12/gdb-multiarch.png)

Perfect! Now we can debug our Arm binary and step through the individual instructions.

FYI: The terminal I use here is [Terminator](https://terminator-gtk3.readthedocs.io/en/latest/) (apt install terminator), which let’s you split the terminal into multiple windows, e.g. CTRL+Shift+O for horizontal split.

For AArch64, you need to run it with qemu-aarch64 and set the target architecture in _gdb-multiarch_ to arm64:

Terminal 1:

```
azeria@ubuntu:~$ qemu-aarch64 -L /usr/aarch64-linux-gnu/ -g 1234 ./hello64
```

Terminal 2:

```
azeria@ubuntu:~$ gdb-multiarch -q --nh -ex 'set architecture arm64' -ex 'file hello64' -ex 'target remote localhost:1234' -ex 'layout split' -ex 'layout regs'
```

For dynamically linked binaries gdb-multiarch will complain about missing libraries. If this happens, run this command in gdb-multiarch and provide the path to the libraries:

```
For AArch64:
(gdb) set solib-search-path /usr/aarch64-linux-gnu/lib/
For AArch32:
(gdb) set solib-search-path /usr/arm-linux-gnueabihf/lib/
```

Happy debugging!
