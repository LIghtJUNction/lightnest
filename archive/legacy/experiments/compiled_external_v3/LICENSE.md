# zlib license and source attribution

The `adler32_z` body is taken from `madler/zlib`, tag `v1.3.1`, file `adler32.c`,
Git blob `04b81d29bad1691878648d7ae1680726abd905e4`.

Source: https://github.com/madler/zlib/blob/v1.3.1/adler32.c
Notice: https://github.com/madler/zlib/blob/v1.3.1/zlib.h

The local source is an **altered standalone extraction**. The function body is
unchanged. Headers are replaced with minimal type definitions, only the normal
division-enabled macros are retained, combine APIs are omitted, and a bounded
benchmark wrapper is added. This is not a new upstream release or a zlib patch.

The upstream distribution notice follows unchanged:

Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

Jean-loup Gailly        Mark Adler
jloup@gzip.org          madler@alumni.caltech.edu
