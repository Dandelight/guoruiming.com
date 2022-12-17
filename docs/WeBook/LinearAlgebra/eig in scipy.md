这篇的起因其实是 `scipy.linalg.eigh` 的几个重载表现不一致，因为难以从数学上解释~~或者说我数学挺菜的不会解释~~，决定从源码中一探究竟。

`eigh` 的完整源码放在了文末，正文中，我们仅保留有关决策的部分

```python
def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=False, eigvals=None, type=1,
         check_finite=True, subset_by_index=None, subset_by_value=None,
         driver=None):
    # Lower 的设置，默认为 'L'
    uplo = 'L' if lower else 'U'
    # 并行用到的，不用管
    _job = 'N' if eigvals_only else 'V'

    # 重要，driver 决定了要使用 LAPACK 中哪个函数
    drv_str = [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]

    # 是否覆写 a，对我们没太大影响
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    # 是否为复数矩阵，决定了是用复数子进程还是用实数子进程
    cplx = True if iscomplexobj(a1) else False
    n = a1.shape[0]

    # subset_by_index 和 subset_by_value 进行一些设置，我们暂时用不到，略

    # 重要，prefix，表示是 HErmitian 还是 SYmmetric，决定调用哪个函数
    pfx = 'he' if cplx else 'sy'

    # decide on the driver if not given
    # first early exit on incompatible choice
    if driver:
        # 略去用户自己指定 driver 的情形
        pass
    # Default driver is evr and gvd
    # 看到这里我们清楚了，如果传入了 b，则使用 evr，否则使用 gvd
    else:
        driver = "evr" if b is None else ("gvx" if subset else "gvd")

    lwork_spec = {
                  'syevd': ['lwork', 'liwork'],
                  'syevr': ['lwork', 'liwork'],
                  'heevd': ['lwork', 'liwork', 'lrwork'],
                  'heevr': ['lwork', 'lrwork', 'liwork'],
                  }

    if b is None:  # Standard problem
        # 拿到 driver
        drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                      [a1])
        clw_args = {'n': n, 'lower': lower}
        if driver == 'evd':
            clw_args.update({'compute_v': 0 if _job == "N" else 1})

        lw = _compute_lwork(drvlw, **clw_args)
        # Multiple lwork vars
        if isinstance(lw, tuple):
            lwork_args = dict(zip(lwork_spec[pfx+driver], lw))
        else:
            lwork_args = {'lwork': lw}

        drv_args.update({'lower': lower, 'compute_v': 0 if _job == "N" else 1})
        # 这里是送入 Fortran subroutine 计算
        w, v, *other_args, info = drv(a=a1, **drv_args, **lwork_args)

    else:  # Generalized problem
        # 差不多的流程，把 evr 换成了 gvd

    # Check if we had a successful exit
    if info == 0:
        if eigvals_only:
            return w
        else:
            return w, v
    else:
        # 一堆报错，不用看了
        raise LinAlgError()
```

可见，如果同时提供了 $A$ 和 $B$ 矩阵，则调用 `evr` 驱动；如果只提供了 $A$ 矩阵，则调用 `gvd` 驱动。问题来了，这俩又是啥？

在 `drv` 中，执行流就离开了 `Python` 进入了 `Fortran` 代码。`scipy` 的数值计算~~和某著名大型商业数学软件一样~~是 Powered by LAPACK 的，所以抱着憧憬的心情我下载了 LAPACK 的代码，

```shell
$ ls SRC/
CMakeLists.txt          dlantp.f                spftri.f
DEPRECATED              dlantr.f                spftrs.f
Makefile                dlanv2.f                spocon.f
VARIANTS                dlaorhr_col_getrfnp.f   spoequ.f
cbbcsd.f                dlaorhr_col_getrfnp2.f  spoequb.f
cbdsqr.f                dlapll.f                sporfs.f
cgbbrd.f                dlapmr.f                sporfsx.f
cgbcon.f                dlapmt.f                sposv.f
cgbequ.f                dlapy2.f                sposvx.f
cgbequb.f               dlapy3.f                sposvxx.f
cgbrfs.f                dlaqgb.f                spotf2.f
cgbrfsx.f               dlaqge.f                spotrf.f
cgbsv.f                 dlaqp2.f                spotrf2.f
# 略去二百来个 .f 文件
```

这么一串没把我吓晕。但奇怪归奇怪，文件命名总有个规律，否则怎么维护怎么使用呢？果然可以很容搜索到 LAPACK subroutine 的命名规范：

> LAPACK 函数以 `XYYZZZ` 命名
>
> `X` 表示数据类型
>
> - `s` 单精度实数
> - `c` 单精度复数
> - `d` 双精度实数
> - `z` 双精度复数
>
> 而 `YY` 表示矩阵的性质
>
> | BD  | bidiagonal                                                                     |
> | --- | ------------------------------------------------------------------------------ |
> | DI  | diagonal                                                                       |
> | GB  | general band                                                                   |
> | GE  | general (i.e., unsymmetric, in some cases rectangular)                         |
> | GG  | general matrices, generalized problem (i.e., a pair of general matrices)       |
> | GT  | general tridiagonal                                                            |
> | HB  | (complex) Hermitian band                                                       |
> | HE  | (complex) Hermitian                                                            |
> | HG  | upper Hessenberg matrix, generalized problem (i.e a Hessenberg and a           |
> |     | triangular matrix)                                                             |
> | HP  | (complex) Hermitian, packed storage                                            |
> | HS  | upper Hessenberg                                                               |
> | OP  | (real) orthogonal, packed storage                                              |
> | OR  | (real) orthogonal                                                              |
> | PB  | symmetric or Hermitian positive definite band                                  |
> | PO  | symmetric or Hermitian positive definite                                       |
> | PP  | symmetric or Hermitian positive definite, packed storage                       |
> | PT  | symmetric or Hermitian positive definite tridiagonal                           |
> | SB  | (real) symmetric band                                                          |
> | SP  | symmetric, packed storage                                                      |
> | ST  | (real) symmetric tridiagonal                                                   |
> | SY  | symmetric                                                                      |
> | TB  | triangular band                                                                |
> | TG  | triangular matrices, generalized problem (i.e., a pair of triangular matrices) |
> | TP  | triangular, packed storage                                                     |
> | TR  | triangular (or in some cases quasi-triangular)                                 |
> | TZ  | trapezoidal                                                                    |
> | UN  | (complex) unitary                                                              |
> | UP  | (complex) unitary, packed storage                                              |
>
> `ZZZ` 表示算法名称，比如 `GVX` 表示 Generalized Eigenvalue Problem （我也不知道 `X` 哪来的），`EVR` 表示 Eigenvalue Problem（我更不知道 `R` 是哪来的了）

但说实在的，我真的看不懂这早期 `Fortran` 代码，只能让各位看官评判了

```fortran
      SUBROUTINE DSYEVR( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU,
     $                   ABSTOL, M, W, Z, LDZ, ISUPPZ, WORK, LWORK,
     $                   IWORK, LIWORK, INFO )
*
*  -- LAPACK driver routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ, RANGE, UPLO
      INTEGER            IL, INFO, IU, LDA, LDZ, LIWORK, LWORK, M, N
      DOUBLE PRECISION   ABSTOL, VL, VU
*     ..
*     .. Array Arguments ..
      INTEGER            ISUPPZ( * ), IWORK( * )
      DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * ), Z( LDZ, * )
*     ..
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ZERO, ONE, TWO
      PARAMETER          ( ZERO = 0.0D+0, ONE = 1.0D+0, TWO = 2.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            ALLEIG, INDEIG, LOWER, LQUERY, VALEIG, WANTZ,
     $                   TRYRAC
      CHARACTER          ORDER
      INTEGER            I, IEEEOK, IINFO, IMAX, INDD, INDDD, INDE,
     $                   INDEE, INDIBL, INDIFL, INDISP, INDIWO, INDTAU,
     $                   INDWK, INDWKN, ISCALE, J, JJ, LIWMIN,
     $                   LLWORK, LLWRKN, LWKOPT, LWMIN, NB, NSPLIT
      DOUBLE PRECISION   ABSTLL, ANRM, BIGNUM, EPS, RMAX, RMIN, SAFMIN,
     $                   SIGMA, SMLNUM, TMP1, VLL, VUU
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ILAENV
      DOUBLE PRECISION   DLAMCH, DLANSY
      EXTERNAL           LSAME, ILAENV, DLAMCH, DLANSY
*     ..
*     .. External Subroutines ..
      EXTERNAL           DCOPY, DORMTR, DSCAL, DSTEBZ, DSTEMR, DSTEIN,
     $                   DSTERF, DSWAP, DSYTRD, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN, SQRT
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      IEEEOK = ILAENV( 10, 'DSYEVR', 'N', 1, 2, 3, 4 )
*
      LOWER = LSAME( UPLO, 'L' )
      WANTZ = LSAME( JOBZ, 'V' )
      ALLEIG = LSAME( RANGE, 'A' )
      VALEIG = LSAME( RANGE, 'V' )
      INDEIG = LSAME( RANGE, 'I' )
*
      LQUERY = ( ( LWORK.EQ.-1 ) .OR. ( LIWORK.EQ.-1 ) )
*
      LWMIN = MAX( 1, 26*N )
      LIWMIN = MAX( 1, 10*N )
*
      INFO = 0
      IF( .NOT.( WANTZ .OR. LSAME( JOBZ, 'N' ) ) ) THEN
         INFO = -1
      ELSE IF( .NOT.( ALLEIG .OR. VALEIG .OR. INDEIG ) ) THEN
         INFO = -2
      ELSE IF( .NOT.( LOWER .OR. LSAME( UPLO, 'U' ) ) ) THEN
         INFO = -3
      ELSE IF( N.LT.0 ) THEN
         INFO = -4
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -6
      ELSE
         IF( VALEIG ) THEN
            IF( N.GT.0 .AND. VU.LE.VL )
     $         INFO = -8
         ELSE IF( INDEIG ) THEN
            IF( IL.LT.1 .OR. IL.GT.MAX( 1, N ) ) THEN
               INFO = -9
            ELSE IF( IU.LT.MIN( N, IL ) .OR. IU.GT.N ) THEN
               INFO = -10
            END IF
         END IF
      END IF
      IF( INFO.EQ.0 ) THEN
         IF( LDZ.LT.1 .OR. ( WANTZ .AND. LDZ.LT.N ) ) THEN
            INFO = -15
         ELSE IF( LWORK.LT.LWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -18
         ELSE IF( LIWORK.LT.LIWMIN .AND. .NOT.LQUERY ) THEN
            INFO = -20
         END IF
      END IF
*
      IF( INFO.EQ.0 ) THEN
         NB = ILAENV( 1, 'DSYTRD', UPLO, N, -1, -1, -1 )
         NB = MAX( NB, ILAENV( 1, 'DORMTR', UPLO, N, -1, -1, -1 ) )
         LWKOPT = MAX( ( NB+1 )*N, LWMIN )
         WORK( 1 ) = LWKOPT
         IWORK( 1 ) = LIWMIN
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSYEVR', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible
*
      M = 0
      IF( N.EQ.0 ) THEN
         WORK( 1 ) = 1
         RETURN
      END IF
*
      IF( N.EQ.1 ) THEN
         WORK( 1 ) = 7
         IF( ALLEIG .OR. INDEIG ) THEN
            M = 1
            W( 1 ) = A( 1, 1 )
         ELSE
            IF( VL.LT.A( 1, 1 ) .AND. VU.GE.A( 1, 1 ) ) THEN
               M = 1
               W( 1 ) = A( 1, 1 )
            END IF
         END IF
         IF( WANTZ ) THEN
            Z( 1, 1 ) = ONE
            ISUPPZ( 1 ) = 1
            ISUPPZ( 2 ) = 1
         END IF
         RETURN
      END IF
*
*     Get machine constants.
*
      SAFMIN = DLAMCH( 'Safe minimum' )
      EPS = DLAMCH( 'Precision' )
      SMLNUM = SAFMIN / EPS
      BIGNUM = ONE / SMLNUM
      RMIN = SQRT( SMLNUM )
      RMAX = MIN( SQRT( BIGNUM ), ONE / SQRT( SQRT( SAFMIN ) ) )
*
*     Scale matrix to allowable range, if necessary.
*
      ISCALE = 0
      ABSTLL = ABSTOL
      IF (VALEIG) THEN
         VLL = VL
         VUU = VU
      END IF
      ANRM = DLANSY( 'M', UPLO, N, A, LDA, WORK )
      IF( ANRM.GT.ZERO .AND. ANRM.LT.RMIN ) THEN
         ISCALE = 1
         SIGMA = RMIN / ANRM
      ELSE IF( ANRM.GT.RMAX ) THEN
         ISCALE = 1
         SIGMA = RMAX / ANRM
      END IF
      IF( ISCALE.EQ.1 ) THEN
         IF( LOWER ) THEN
            DO 10 J = 1, N
               CALL DSCAL( N-J+1, SIGMA, A( J, J ), 1 )
   10       CONTINUE
         ELSE
            DO 20 J = 1, N
               CALL DSCAL( J, SIGMA, A( 1, J ), 1 )
   20       CONTINUE
         END IF
         IF( ABSTOL.GT.0 )
     $      ABSTLL = ABSTOL*SIGMA
         IF( VALEIG ) THEN
            VLL = VL*SIGMA
            VUU = VU*SIGMA
         END IF
      END IF

*     Initialize indices into workspaces.  Note: The IWORK indices are
*     used only if DSTERF or DSTEMR fail.

*     WORK(INDTAU:INDTAU+N-1) stores the scalar factors of the
*     elementary reflectors used in DSYTRD.
      INDTAU = 1
*     WORK(INDD:INDD+N-1) stores the tridiagonal's diagonal entries.
      INDD = INDTAU + N
*     WORK(INDE:INDE+N-1) stores the off-diagonal entries of the
*     tridiagonal matrix from DSYTRD.
      INDE = INDD + N
*     WORK(INDDD:INDDD+N-1) is a copy of the diagonal entries over
*     -written by DSTEMR (the DSTERF path copies the diagonal to W).
      INDDD = INDE + N
*     WORK(INDEE:INDEE+N-1) is a copy of the off-diagonal entries over
*     -written while computing the eigenvalues in DSTERF and DSTEMR.
      INDEE = INDDD + N
*     INDWK is the starting offset of the left-over workspace, and
*     LLWORK is the remaining workspace size.
      INDWK = INDEE + N
      LLWORK = LWORK - INDWK + 1

*     IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in DSTEBZ and
*     stores the block indices of each of the M<=N eigenvalues.
      INDIBL = 1
*     IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in DSTEBZ and
*     stores the starting and finishing indices of each block.
      INDISP = INDIBL + N
*     IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
*     that corresponding to eigenvectors that fail to converge in
*     DSTEIN.  This information is discarded; if any fail, the driver
*     returns INFO > 0.
      INDIFL = INDISP + N
*     INDIWO is the offset of the remaining integer workspace.
      INDIWO = INDIFL + N

*
*     Call DSYTRD to reduce symmetric matrix to tridiagonal form.
*
      CALL DSYTRD( UPLO, N, A, LDA, WORK( INDD ), WORK( INDE ),
     $             WORK( INDTAU ), WORK( INDWK ), LLWORK, IINFO )
*
*     If all eigenvalues are desired
*     then call DSTERF or DSTEMR and DORMTR.
*
      IF( ( ALLEIG .OR. ( INDEIG .AND. IL.EQ.1 .AND. IU.EQ.N ) ) .AND.
     $    IEEEOK.EQ.1 ) THEN
         IF( .NOT.WANTZ ) THEN
            CALL DCOPY( N, WORK( INDD ), 1, W, 1 )
            CALL DCOPY( N-1, WORK( INDE ), 1, WORK( INDEE ), 1 )
            CALL DSTERF( N, W, WORK( INDEE ), INFO )
         ELSE
            CALL DCOPY( N-1, WORK( INDE ), 1, WORK( INDEE ), 1 )
            CALL DCOPY( N, WORK( INDD ), 1, WORK( INDDD ), 1 )
*
            IF (ABSTOL .LE. TWO*N*EPS) THEN
               TRYRAC = .TRUE.
            ELSE
               TRYRAC = .FALSE.
            END IF
            CALL DSTEMR( JOBZ, 'A', N, WORK( INDDD ), WORK( INDEE ),
     $                   VL, VU, IL, IU, M, W, Z, LDZ, N, ISUPPZ,
     $                   TRYRAC, WORK( INDWK ), LWORK, IWORK, LIWORK,
     $                   INFO )
*
*
*
*        Apply orthogonal matrix used in reduction to tridiagonal
*        form to eigenvectors returned by DSTEMR.
*
            IF( WANTZ .AND. INFO.EQ.0 ) THEN
               INDWKN = INDE
               LLWRKN = LWORK - INDWKN + 1
               CALL DORMTR( 'L', UPLO, 'N', N, M, A, LDA,
     $                      WORK( INDTAU ), Z, LDZ, WORK( INDWKN ),
     $                      LLWRKN, IINFO )
            END IF
         END IF
*
*
         IF( INFO.EQ.0 ) THEN
*           Everything worked.  Skip DSTEBZ/DSTEIN.  IWORK(:) are
*           undefined.
            M = N
            GO TO 30
         END IF
         INFO = 0
      END IF
*
*     Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN.
*     Also call DSTEBZ and DSTEIN if DSTEMR fails.
*
      IF( WANTZ ) THEN
         ORDER = 'B'
      ELSE
         ORDER = 'E'
      END IF

      CALL DSTEBZ( RANGE, ORDER, N, VLL, VUU, IL, IU, ABSTLL,
     $             WORK( INDD ), WORK( INDE ), M, NSPLIT, W,
     $             IWORK( INDIBL ), IWORK( INDISP ), WORK( INDWK ),
     $             IWORK( INDIWO ), INFO )
*
      IF( WANTZ ) THEN
         CALL DSTEIN( N, WORK( INDD ), WORK( INDE ), M, W,
     $                IWORK( INDIBL ), IWORK( INDISP ), Z, LDZ,
     $                WORK( INDWK ), IWORK( INDIWO ), IWORK( INDIFL ),
     $                INFO )
*
*        Apply orthogonal matrix used in reduction to tridiagonal
*        form to eigenvectors returned by DSTEIN.
*
         INDWKN = INDE
         LLWRKN = LWORK - INDWKN + 1
         CALL DORMTR( 'L', UPLO, 'N', N, M, A, LDA, WORK( INDTAU ), Z,
     $                LDZ, WORK( INDWKN ), LLWRKN, IINFO )
      END IF
*
*     If matrix was scaled, then rescale eigenvalues appropriately.
*
*  Jump here if DSTEMR/DSTEIN succeeded.
   30 CONTINUE
      IF( ISCALE.EQ.1 ) THEN
         IF( INFO.EQ.0 ) THEN
            IMAX = M
         ELSE
            IMAX = INFO - 1
         END IF
         CALL DSCAL( IMAX, ONE / SIGMA, W, 1 )
      END IF
*
*     If eigenvalues are not in order, then sort them, along with
*     eigenvectors.  Note: We do not sort the IFAIL portion of IWORK.
*     It may not be initialized (if DSTEMR/DSTEIN succeeded), and we do
*     not return this detailed information to the user.
*
      IF( WANTZ ) THEN
         DO 50 J = 1, M - 1
            I = 0
            TMP1 = W( J )
            DO 40 JJ = J + 1, M
               IF( W( JJ ).LT.TMP1 ) THEN
                  I = JJ
                  TMP1 = W( JJ )
               END IF
   40       CONTINUE
*
            IF( I.NE.0 ) THEN
               W( I ) = W( J )
               W( J ) = TMP1
               CALL DSWAP( N, Z( 1, I ), 1, Z( 1, J ), 1 )
            END IF
   50    CONTINUE
      END IF
*
*     Set WORK(1) to optimal workspace size.
*
      WORK( 1 ) = LWKOPT
      IWORK( 1 ) = LIWMIN
*
      RETURN
*
*     End of DSYEVR
*
      END
```

```fortran
      SUBROUTINE DSYGVX( ITYPE, JOBZ, RANGE, UPLO, N, A, LDA, B, LDB,
     $                   VL, VU, IL, IU, ABSTOL, M, W, Z, LDZ, WORK,
     $                   LWORK, IWORK, IFAIL, INFO )
*
*  -- LAPACK driver routine --
*  -- LAPACK is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*
*     .. Scalar Arguments ..
      CHARACTER          JOBZ, RANGE, UPLO
      INTEGER            IL, INFO, ITYPE, IU, LDA, LDB, LDZ, LWORK, M, N
      DOUBLE PRECISION   ABSTOL, VL, VU
*     ..
*     .. Array Arguments ..
      INTEGER            IFAIL( * ), IWORK( * )
      DOUBLE PRECISION   A( LDA, * ), B( LDB, * ), W( * ), WORK( * ),
     $                   Z( LDZ, * )
*     ..
*
* =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION   ONE
      PARAMETER          ( ONE = 1.0D+0 )
*     ..
*     .. Local Scalars ..
      LOGICAL            ALLEIG, INDEIG, LQUERY, UPPER, VALEIG, WANTZ
      CHARACTER          TRANS
      INTEGER            LWKMIN, LWKOPT, NB
*     ..
*     .. External Functions ..
      LOGICAL            LSAME
      INTEGER            ILAENV
      EXTERNAL           LSAME, ILAENV
*     ..
*     .. External Subroutines ..
      EXTERNAL           DPOTRF, DSYEVX, DSYGST, DTRMM, DTRSM, XERBLA
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC          MAX, MIN
*     ..
*     .. Executable Statements ..
*
*     Test the input parameters.
*
      UPPER = LSAME( UPLO, 'U' )
      WANTZ = LSAME( JOBZ, 'V' )
      ALLEIG = LSAME( RANGE, 'A' )
      VALEIG = LSAME( RANGE, 'V' )
      INDEIG = LSAME( RANGE, 'I' )
      LQUERY = ( LWORK.EQ.-1 )
*
      INFO = 0
      IF( ITYPE.LT.1 .OR. ITYPE.GT.3 ) THEN
         INFO = -1
      ELSE IF( .NOT.( WANTZ .OR. LSAME( JOBZ, 'N' ) ) ) THEN
         INFO = -2
      ELSE IF( .NOT.( ALLEIG .OR. VALEIG .OR. INDEIG ) ) THEN
         INFO = -3
      ELSE IF( .NOT.( UPPER .OR. LSAME( UPLO, 'L' ) ) ) THEN
         INFO = -4
      ELSE IF( N.LT.0 ) THEN
         INFO = -5
      ELSE IF( LDA.LT.MAX( 1, N ) ) THEN
         INFO = -7
      ELSE IF( LDB.LT.MAX( 1, N ) ) THEN
         INFO = -9
      ELSE
         IF( VALEIG ) THEN
            IF( N.GT.0 .AND. VU.LE.VL )
     $         INFO = -11
         ELSE IF( INDEIG ) THEN
            IF( IL.LT.1 .OR. IL.GT.MAX( 1, N ) ) THEN
               INFO = -12
            ELSE IF( IU.LT.MIN( N, IL ) .OR. IU.GT.N ) THEN
               INFO = -13
            END IF
         END IF
      END IF
      IF (INFO.EQ.0) THEN
         IF (LDZ.LT.1 .OR. (WANTZ .AND. LDZ.LT.N)) THEN
            INFO = -18
         END IF
      END IF
*
      IF( INFO.EQ.0 ) THEN
         LWKMIN = MAX( 1, 8*N )
         NB = ILAENV( 1, 'DSYTRD', UPLO, N, -1, -1, -1 )
         LWKOPT = MAX( LWKMIN, ( NB + 3 )*N )
         WORK( 1 ) = LWKOPT
*
         IF( LWORK.LT.LWKMIN .AND. .NOT.LQUERY ) THEN
            INFO = -20
         END IF
      END IF
*
      IF( INFO.NE.0 ) THEN
         CALL XERBLA( 'DSYGVX', -INFO )
         RETURN
      ELSE IF( LQUERY ) THEN
         RETURN
      END IF
*
*     Quick return if possible
*
      M = 0
      IF( N.EQ.0 ) THEN
         RETURN
      END IF
*
*     Form a Cholesky factorization of B.
*
      CALL DPOTRF( UPLO, N, B, LDB, INFO )
      IF( INFO.NE.0 ) THEN
         INFO = N + INFO
         RETURN
      END IF
*
*     Transform problem to standard eigenvalue problem and solve.
*
      CALL DSYGST( ITYPE, UPLO, N, A, LDA, B, LDB, INFO )
      CALL DSYEVX( JOBZ, RANGE, UPLO, N, A, LDA, VL, VU, IL, IU, ABSTOL,
     $             M, W, Z, LDZ, WORK, LWORK, IWORK, IFAIL, INFO )
*
      IF( WANTZ ) THEN
*
*        Backtransform eigenvectors to the original problem.
*
         IF( INFO.GT.0 )
     $      M = INFO - 1
         IF( ITYPE.EQ.1 .OR. ITYPE.EQ.2 ) THEN
*
*           For A*x=(lambda)*B*x and A*B*x=(lambda)*x;
*           backtransform eigenvectors: x = inv(L)**T*y or inv(U)*y
*
            IF( UPPER ) THEN
               TRANS = 'N'
            ELSE
               TRANS = 'T'
            END IF
*
            CALL DTRSM( 'Left', UPLO, TRANS, 'Non-unit', N, M, ONE, B,
     $                  LDB, Z, LDZ )
*
         ELSE IF( ITYPE.EQ.3 ) THEN
*
*           For B*A*x=(lambda)*x;
*           backtransform eigenvectors: x = L*y or U**T*y
*
            IF( UPPER ) THEN
               TRANS = 'T'
            ELSE
               TRANS = 'N'
            END IF
*
            CALL DTRMM( 'Left', UPLO, TRANS, 'Non-unit', N, M, ONE, B,
     $                  LDB, Z, LDZ )
         END IF
      END IF
*
*     Set WORK(1) to optimal workspace size.
*
      WORK( 1 ) = LWKOPT
*
      RETURN
*
*     End of DSYGVX
*
      END
```

```python
# https://github.com/scipy/scipy/blob/46811bed97a375ecf700174cf86675e10fb57a57/scipy/linalg/_decomp.py#L117
def eig(a, b=None, left=False, right=True, overwrite_a=False,
        overwrite_b=False, check_finite=True, homogeneous_eigvals=False):
    """
    Solve an ordinary or generalized eigenvalue problem of a square matrix.

    Find eigenvalues w and right or left eigenvectors of a general matrix::

        a   vr[:,i] = w[i]        b   vr[:,i]
        a.H vl[:,i] = w[i].conj() b.H vl[:,i]

    where ``.H`` is the Hermitian conjugation.

    Parameters
    ----------
    a : (M, M) array_like
        A complex or real matrix whose eigenvalues and eigenvectors
        will be computed.
    b : (M, M) array_like, optional
        Right-hand side matrix in a generalized eigenvalue problem.
        Default is None, identity matrix is assumed.
    left : bool, optional
        Whether to calculate and return left eigenvectors.  Default is False.
    right : bool, optional
        Whether to calculate and return right eigenvectors.  Default is True.
    overwrite_a : bool, optional
        Whether to overwrite `a`; may improve performance.  Default is False.
    overwrite_b : bool, optional
        Whether to overwrite `b`; may improve performance.  Default is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    homogeneous_eigvals : bool, optional
        If True, return the eigenvalues in homogeneous coordinates.
        In this case ``w`` is a (2, M) array so that::

            w[1,i] a vr[:,i] = w[0,i] b vr[:,i]

        Default is False.

    Returns
    -------
    w : (M,) or (2, M) double or complex ndarray
        The eigenvalues, each repeated according to its
        multiplicity. The shape is (M,) unless
        ``homogeneous_eigvals=True``.
    vl : (M, M) double or complex ndarray
        The normalized left eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column vl[:,i]. Only returned if ``left=True``.
    vr : (M, M) double or complex ndarray
        The normalized right eigenvector corresponding to the eigenvalue
        ``w[i]`` is the column ``vr[:,i]``.  Only returned if ``right=True``.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigvals : eigenvalues of general arrays
    eigh : Eigenvalues and right eigenvectors for symmetric/Hermitian arrays.
    eig_banded : eigenvalues and right eigenvectors for symmetric/Hermitian
        band matrices
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a)
    array([0.+1.j, 0.-1.j])

    >>> b = np.array([[0., 1.], [1., 1.]])
    >>> linalg.eigvals(a, b)
    array([ 1.+0.j, -1.+0.j])

    >>> a = np.array([[3., 0., 0.], [0., 8., 0.], [0., 0., 7.]])
    >>> linalg.eigvals(a, homogeneous_eigvals=True)
    array([[3.+0.j, 8.+0.j, 7.+0.j],
           [1.+0.j, 1.+0.j, 1.+0.j]])

    >>> a = np.array([[0., -1.], [1., 0.]])
    >>> linalg.eigvals(a) == linalg.eig(a)[0]
    array([ True,  True])
    >>> linalg.eig(a, left=True, right=False)[1] # normalized left eigenvector
    array([[-0.70710678+0.j        , -0.70710678-0.j        ],
           [-0.        +0.70710678j, -0.        -0.70710678j]])
    >>> linalg.eig(a, left=False, right=True)[1] # normalized right eigenvector
    array([[0.70710678+0.j        , 0.70710678-0.j        ],
           [0.        -0.70710678j, 0.        +0.70710678j]])



    """
    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square matrix')
        if b1.shape != a1.shape:
            raise ValueError('a and b must have the same shape')
        return _geneig(a1, b1, left, right, overwrite_a, overwrite_b,
                       homogeneous_eigvals)

    geev, geev_lwork = get_lapack_funcs(('geev', 'geev_lwork'), (a1,))
    compute_vl, compute_vr = left, right

    lwork = _compute_lwork(geev_lwork, a1.shape[0],
                           compute_vl=compute_vl,
                           compute_vr=compute_vr)

    if geev.typecode in 'cz':
        w, vl, vr, info = geev(a1, lwork=lwork,
                               compute_vl=compute_vl,
                               compute_vr=compute_vr,
                               overwrite_a=overwrite_a)
        w = _make_eigvals(w, None, homogeneous_eigvals)
    else:
        wr, wi, vl, vr, info = geev(a1, lwork=lwork,
                                    compute_vl=compute_vl,
                                    compute_vr=compute_vr,
                                    overwrite_a=overwrite_a)
        t = {'f': 'F', 'd': 'D'}[wr.dtype.char]
        w = wr + _I * wi
        w = _make_eigvals(w, None, homogeneous_eigvals)

    _check_info(info, 'eig algorithm (geev)',
                positive='did not converge (only eigenvalues '
                         'with order >= %d have converged)')

    only_real = numpy.all(w.imag == 0.0)
    if not (geev.typecode in 'cz' or only_real):
        t = w.dtype.char
        if left:
            vl = _make_complex_eigvecs(w, vl, t)
        if right:
            vr = _make_complex_eigvecs(w, vr, t)
    if not (left or right):
        return w
    if left:
        if right:
            return w, vl, vr
        return w, vl
    return w, vr


def eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False,
         overwrite_b=False, turbo=False, eigvals=None, type=1,
         check_finite=True, subset_by_index=None, subset_by_value=None,
         driver=None):
    """
    Solve a standard or generalized eigenvalue problem for a complex
    Hermitian or real symmetric matrix.

    Find eigenvalues array ``w`` and optionally eigenvectors array ``v`` of
    array ``a``, where ``b`` is positive definite such that for every
    eigenvalue λ (i-th entry of w) and its eigenvector ``vi`` (i-th column of
    ``v``) satisfies::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    In the standard problem, ``b`` is assumed to be the identity matrix.

    Parameters
    ----------
    a : (M, M) array_like
        A complex Hermitian or real symmetric matrix whose eigenvalues and
        eigenvectors will be computed.
    b : (M, M) array_like, optional
        A complex Hermitian or real symmetric definite positive matrix in.
        If omitted, identity matrix is assumed.
    lower : bool, optional
        Whether the pertinent array data is taken from the lower or upper
        triangle of ``a`` and, if applicable, ``b``. (Default: lower)
    eigvals_only : bool, optional
        Whether to calculate only eigenvalues and no eigenvectors.
        (Default: both are calculated)
    subset_by_index : iterable, optional
        If provided, this two-element iterable defines the start and the end
        indices of the desired eigenvalues (ascending order and 0-indexed).
        To return only the second smallest to fifth smallest eigenvalues,
        ``[1, 4]`` is used. ``[n-3, n-1]`` returns the largest three. Only
        available with "evr", "evx", and "gvx" drivers. The entries are
        directly converted to integers via ``int()``.
    subset_by_value : iterable, optional
        If provided, this two-element iterable defines the half-open interval
        ``(a, b]`` that, if any, only the eigenvalues between these values
        are returned. Only available with "evr", "evx", and "gvx" drivers. Use
        ``np.inf`` for the unconstrained ends.
    driver : str, optional
        Defines which LAPACK driver should be used. Valid options are "ev",
        "evd", "evr", "evx" for standard problems and "gv", "gvd", "gvx" for
        generalized (where b is not None) problems. See the Notes section.
        The default for standard problems is "evr". For generalized problems,
        "gvd" is used for full set, and "gvx" for subset requested cases.
    type : int, optional
        For the generalized problems, this keyword specifies the problem type
        to be solved for ``w`` and ``v`` (only takes 1, 2, 3 as possible
        inputs)::

            1 =>     a @ v = w @ b @ v
            2 => a @ b @ v = w @ v
            3 => b @ a @ v = w @ v

        This keyword is ignored for standard problems.
    overwrite_a : bool, optional
        Whether to overwrite data in ``a`` (may improve performance). Default
        is False.
    overwrite_b : bool, optional
        Whether to overwrite data in ``b`` (may improve performance). Default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    turbo : bool, optional, deprecated
            .. deprecated:: 1.5.0
                `eigh` keyword argument `turbo` is deprecated in favour of
                ``driver=gvd`` keyword instead and will be removed in SciPy
                1.12.0.
    eigvals : tuple (lo, hi), optional, deprecated
            .. deprecated:: 1.5.0
                `eigh` keyword argument `eigvals` is deprecated in favour of
                `subset_by_index` keyword instead and will be removed in SciPy
                1.12.0.

    Returns
    -------
    w : (N,) ndarray
        The N (1<=N<=M) selected eigenvalues, in ascending order, each
        repeated according to its multiplicity.
    v : (M, N) ndarray
        (if ``eigvals_only == False``)

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge, an error occurred, or
        b matrix is not definite positive. Note that if input matrices are
        not symmetric or Hermitian, no error will be reported but results will
        be wrong.

    See Also
    --------
    eigvalsh : eigenvalues of symmetric or Hermitian arrays
    eig : eigenvalues and right eigenvectors for non-symmetric arrays
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Notes
    -----
    This function does not check the input array for being Hermitian/symmetric
    in order to allow for representing arrays with only their upper/lower
    triangular parts. Also, note that even though not taken into account,
    finiteness check applies to the whole array and unaffected by "lower"
    keyword.

    This function uses LAPACK drivers for computations in all possible keyword
    combinations, prefixed with ``sy`` if arrays are real and ``he`` if
    complex, e.g., a float array with "evr" driver is solved via
    "syevr", complex arrays with "gvx" driver problem is solved via "hegvx"
    etc.

    As a brief summary, the slowest and the most robust driver is the
    classical ``<sy/he>ev`` which uses symmetric QR. ``<sy/he>evr`` is seen as
    the optimal choice for the most general cases. However, there are certain
    occasions that ``<sy/he>evd`` computes faster at the expense of more
    memory usage. ``<sy/he>evx``, while still being faster than ``<sy/he>ev``,
    often performs worse than the rest except when very few eigenvalues are
    requested for large arrays though there is still no performance guarantee.


    For the generalized problem, normalization with respect to the given
    type argument::

            type 1 and 3 :      v.conj().T @ a @ v = w
            type 2       : inv(v).conj().T @ a @ inv(v) = w

            type 1 or 2  :      v.conj().T @ b @ v  = I
            type 3       : v.conj().T @ inv(b) @ v  = I


    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigh
    >>> A = np.array([[6, 3, 1, 5], [3, 0, 5, 1], [1, 5, 6, 2], [5, 1, 2, 2]])
    >>> w, v = eigh(A)
    >>> np.allclose(A @ v - v @ np.diag(w), np.zeros((4, 4)))
    True

    Request only the eigenvalues

    >>> w = eigh(A, eigvals_only=True)

    Request eigenvalues that are less than 10.

    >>> A = np.array([[34, -4, -10, -7, 2],
    ...               [-4, 7, 2, 12, 0],
    ...               [-10, 2, 44, 2, -19],
    ...               [-7, 12, 2, 79, -34],
    ...               [2, 0, -19, -34, 29]])
    >>> eigh(A, eigvals_only=True, subset_by_value=[-np.inf, 10])
    array([6.69199443e-07, 9.11938152e+00])

    Request the second smallest eigenvalue and its eigenvector

    >>> w, v = eigh(A, subset_by_index=[1, 1])
    >>> w
    array([9.11938152])
    >>> v.shape  # only a single column is returned
    (5, 1)

    """
    if turbo:
        warnings.warn("Keyword argument 'turbo' is deprecated in favour of '"
                      "driver=gvd' keyword instead and will be removed in "
                      "SciPy 1.12.0.",
                      DeprecationWarning, stacklevel=2)
    if eigvals:
        warnings.warn("Keyword argument 'eigvals' is deprecated in favour of "
                      "'subset_by_index' keyword instead and will be removed "
                      "in SciPy 1.12.0.",
                      DeprecationWarning, stacklevel=2)

    # set lower
    uplo = 'L' if lower else 'U'
    # Set job for Fortran routines
    _job = 'N' if eigvals_only else 'V'

    drv_str = [None, "ev", "evd", "evr", "evx", "gv", "gvd", "gvx"]
    if driver not in drv_str:
        raise ValueError('"{}" is unknown. Possible values are "None", "{}".'
                         ''.format(driver, '", "'.join(drv_str[1:])))

    a1 = _asarray_validated(a, check_finite=check_finite)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square "a" matrix')
    overwrite_a = overwrite_a or (_datacopied(a1, a))
    cplx = True if iscomplexobj(a1) else False
    n = a1.shape[0]
    drv_args = {'overwrite_a': overwrite_a}

    if b is not None:
        b1 = _asarray_validated(b, check_finite=check_finite)
        overwrite_b = overwrite_b or _datacopied(b1, b)
        if len(b1.shape) != 2 or b1.shape[0] != b1.shape[1]:
            raise ValueError('expected square "b" matrix')

        if b1.shape != a1.shape:
            raise ValueError("wrong b dimensions {}, should "
                             "be {}".format(b1.shape, a1.shape))

        if type not in [1, 2, 3]:
            raise ValueError('"type" keyword only accepts 1, 2, and 3.')

        cplx = True if iscomplexobj(b1) else (cplx or False)
        drv_args.update({'overwrite_b': overwrite_b, 'itype': type})

    # backwards-compatibility handling
    subset_by_index = subset_by_index if (eigvals is None) else eigvals

    subset = (subset_by_index is not None) or (subset_by_value is not None)

    # Both subsets can't be given
    if subset_by_index and subset_by_value:
        raise ValueError('Either index or value subset can be requested.')

    # Take turbo into account if all conditions are met otherwise ignore
    if turbo and b is not None:
        driver = 'gvx' if subset else 'gvd'

    # Check indices if given
    if subset_by_index:
        lo, hi = [int(x) for x in subset_by_index]
        if not (0 <= lo <= hi < n):
            raise ValueError('Requested eigenvalue indices are not valid. '
                             'Valid range is [0, {}] and start <= end, but '
                             'start={}, end={} is given'.format(n-1, lo, hi))
        # fortran is 1-indexed
        drv_args.update({'range': 'I', 'il': lo + 1, 'iu': hi + 1})

    if subset_by_value:
        lo, hi = subset_by_value
        if not (-inf <= lo < hi <= inf):
            raise ValueError('Requested eigenvalue bounds are not valid. '
                             'Valid range is (-inf, inf) and low < high, but '
                             'low={}, high={} is given'.format(lo, hi))

        drv_args.update({'range': 'V', 'vl': lo, 'vu': hi})

    # fix prefix for lapack routines
    pfx = 'he' if cplx else 'sy'

    # decide on the driver if not given
    # first early exit on incompatible choice
    if driver:
        if b is None and (driver in ["gv", "gvd", "gvx"]):
            raise ValueError('{} requires input b array to be supplied '
                             'for generalized eigenvalue problems.'
                             ''.format(driver))
        if (b is not None) and (driver in ['ev', 'evd', 'evr', 'evx']):
            raise ValueError('"{}" does not accept input b array '
                             'for standard eigenvalue problems.'
                             ''.format(driver))
        if subset and (driver in ["ev", "evd", "gv", "gvd"]):
            raise ValueError('"{}" cannot compute subsets of eigenvalues'
                             ''.format(driver))

    # Default driver is evr and gvd
    else:
        driver = "evr" if b is None else ("gvx" if subset else "gvd")

    lwork_spec = {
                  'syevd': ['lwork', 'liwork'],
                  'syevr': ['lwork', 'liwork'],
                  'heevd': ['lwork', 'liwork', 'lrwork'],
                  'heevr': ['lwork', 'lrwork', 'liwork'],
                  }

    if b is None:  # Standard problem
        drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                      [a1])
        clw_args = {'n': n, 'lower': lower}
        if driver == 'evd':
            clw_args.update({'compute_v': 0 if _job == "N" else 1})

        lw = _compute_lwork(drvlw, **clw_args)
        # Multiple lwork vars
        if isinstance(lw, tuple):
            lwork_args = dict(zip(lwork_spec[pfx+driver], lw))
        else:
            lwork_args = {'lwork': lw}

        drv_args.update({'lower': lower, 'compute_v': 0 if _job == "N" else 1})
        w, v, *other_args, info = drv(a=a1, **drv_args, **lwork_args)

    else:  # Generalized problem
        # 'gvd' doesn't have lwork query
        if driver == "gvd":
            drv = get_lapack_funcs(pfx + "gvd", [a1, b1])
            lwork_args = {}
        else:
            drv, drvlw = get_lapack_funcs((pfx + driver, pfx+driver+'_lwork'),
                                          [a1, b1])
            # generalized drivers use uplo instead of lower
            lw = _compute_lwork(drvlw, n, uplo=uplo)
            lwork_args = {'lwork': lw}

        drv_args.update({'uplo': uplo, 'jobz': _job})

        w, v, *other_args, info = drv(a=a1, b=b1, **drv_args, **lwork_args)

    # m is always the first extra argument
    w = w[:other_args[0]] if subset else w
    v = v[:, :other_args[0]] if (subset and not eigvals_only) else v

    # Check if we had a  successful exit
    if info == 0:
        if eigvals_only:
            return w
        else:
            return w, v
    else:
        if info < -1:
            raise LinAlgError('Illegal value in argument {} of internal {}'
                              ''.format(-info, drv.typecode + pfx + driver))
        elif info > n:
            raise LinAlgError('The leading minor of order {} of B is not '
                              'positive definite. The factorization of B '
                              'could not be completed and no eigenvalues '
                              'or eigenvectors were computed.'.format(info-n))
        else:
            drv_err = {'ev': 'The algorithm failed to converge; {} '
                             'off-diagonal elements of an intermediate '
                             'tridiagonal form did not converge to zero.',
                       'evx': '{} eigenvectors failed to converge.',
                       'evd': 'The algorithm failed to compute an eigenvalue '
                              'while working on the submatrix lying in rows '
                              'and columns {0}/{1} through mod({0},{1}).',
                       'evr': 'Internal Error.'
                       }
            if driver in ['ev', 'gv']:
                msg = drv_err['ev'].format(info)
            elif driver in ['evx', 'gvx']:
                msg = drv_err['evx'].format(info)
            elif driver in ['evd', 'gvd']:
                if eigvals_only:
                    msg = drv_err['ev'].format(info)
                else:
                    msg = drv_err['evd'].format(info, n+1)
            else:
                msg = drv_err['evr']

            raise LinAlgError(msg)
```
