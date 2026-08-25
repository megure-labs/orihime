# HIP source provenance

The HIP implementation in this release was imported from Megure Labs' clean,
committed private predecessor tree at commit
`a49269fc27f77259450276dce2ad1f3c77dab53c`. The source checkout had no local
modifications. Later uncommitted autotuning work from a separate AMD machine
was deliberately excluded.

No HIP kernel was authored, regenerated, optimized, or algorithmically changed
while preparing this public cut. Each of the 40 files below was copied and
subjected to two literal, case-sensitive substitutions:

- the predecessor's uppercase project identifier became `ORIHIME`; and
- the predecessor's lowercase project identifier became `orihime`; and
- trailing horizontal whitespace was removed from the imported text files.

Nine `torch_hip.cpp` files received binding-only compatibility repairs. Eight
imported wrappers predated the public namespaced operator surface, so their
wrapper and registration tails were mechanically synchronized from the tested
CUDA equivalents. Function suffixes and low-level calls were mapped to the
existing HIP implementations; the MAS tensor-temperature adapter was likewise
copied from its CUDA counterpart, with only validation and low-level calls
redirected to HIP. Namespace placement and one wrapper-name collision were
adjusted where the older HIP file layout differed. `src/nw_affine/torch_hip.cpp`
already had the namespaced surface, but accidentally declared and returned
three results where its schema and CPU/CUDA counterparts return `(score,
grad_gap_open, grad_gap_ext, grad_temperature)`; it now returns the
already-computed temperature-gradient tensor at tuple index 4. No HIP kernel
body or algorithm changed.

All other imported bytes match that deterministic rename-and-whitespace
transformation. The first digest is the private committed source; the second
is the resulting public Orihime file. Both are SHA-256.

| File | Source SHA-256 | Public SHA-256 |
| --- | --- | --- |
| `src/cky/kernels.hip` | `bfedeb5996b6b18a5acbc10d73210d3a0f361afb170a701330cc1f1414d661ee` | `2967ce73b6fba5193dea9a1d60fe32cb327f8d1d6a8877bb6ba593f4571e2c07` |
| `src/cky/kernels.hiph` | `3d42678f6bc2e14110ef3e988efc0cf41d9bb0ae818e4b581a1975ec71d10ac9` | `3d42678f6bc2e14110ef3e988efc0cf41d9bb0ae818e4b581a1975ec71d10ac9` |
| `src/cky/torch_hip.cpp` | `e024696a2797a6dc0bd5d4c7231d82a7dc23adc1a6b7a366f6aa732c069b6078` | `0f2b31a5bbeea6598a688349161bf191b610411c9dfb2721242477103a4fa2c1` |
| `src/common/hip_utils.h` | `57dcee0c848fd63c70a0bd0a2d5233702d80c473c1485fe511a65f299184e498` | `1e9dde3d5f1228502fe0715eda3cbbab789351fc28082be1ac8a622dcc6fd51c` |
| `src/common/numerics.hiph` | `8e5b89ea152ca4e31620d4ee40b29b324c8fd874028ce3a14685c7a57a0a2066` | `b7adf87ddc08865ff1e589154db0bcef12bec623f9daef626679fa176640c39b` |
| `src/common/reduce.hiph` | `0c8e8627ca0f4710bd654a30eeb3c43f835b12880ba1713e1eae5cf2e4668d04` | `1e9b21e19e23c5351a24d38766ba96dc80b93df95e604c9b889241c01cc5cd95` |
| `src/common/softmax.hiph` | `0a86cf341a92478ebd61dc8fef5e85487c2842c0051d2f744bb800c3a6268508` | `9f34985c472964d90603958678ffd06a966ea4ea5f585671e2bb67cd53fec3d7` |
| `src/damerau/kernels.hip` | `f97803305b89e876d9737f2ba8909a5f2f79b17706d7dc7c2d11f9f7834cabd9` | `83214cb4bcf37827b70759c4ae7f82ec10f5d58d8690e3ae6d5e9044cde5aa28` |
| `src/damerau/kernels.hiph` | `b0de2ceb2c9ad4c48bec933b943afdf4140fca6bb68956ba03bcfe05bbdfbedb` | `40ca8448b20fe8e30ba54e36c7c834bbf6a040878b625ea040be0e67a9b9edc5` |
| `src/damerau/torch_hip.cpp` | `528b72944c2316ebe297c234bde907306d35dcdf1275ec09883ddb0bddcd055b` | `0ee7cddda09c232aa251557fa5f8b1c613fce3eabb7670faab10aeff1560afdb` |
| `src/dtw/kernels.hip` | `24435b5b477c5e4a15164328e5c1b46f9acb22e750d28486618ca34e361aff02` | `7e0317c70fd48366df314a5b8b5df60d75b68286f352b929c038432caa9ec99c` |
| `src/dtw/kernels.hiph` | `dadd22d000306a6dfc6bbb4686a9f119995c943a1fbebb36c506b142a2ae6c67` | `dadd22d000306a6dfc6bbb4686a9f119995c943a1fbebb36c506b142a2ae6c67` |
| `src/dtw/torch_hip.cpp` | `06358fd79a2ef92086ec6f61ef80918c484d1f90f4a194927756d53c930fceb7` | `06c66a6e9ed64d8504b0913e51ec4882a0792d62545b64ba1dbb460de3abe3db` |
| `src/eisner/kernels.hip` | `dfb0b26d4d605cb36a64bbca8d7d58942ba34c24986f4b7881b01c9a2f5487ec` | `f76335e8142085d6a968c0d5015bd4d9b078601c86f6088f3e95dfee31f49d81` |
| `src/eisner/kernels.hiph` | `1ef2f3676190f1b5b31151753b940190540e6c3675a0bb4b7b625826bc1547a7` | `6c4ff6f4678fe067489b78ec17c332a2fe6245ddecee3363cb44c097c45f7949` |
| `src/eisner/torch_hip.cpp` | `07dfac10eff199049b6c692c1789dee4f79bf5265caf84e3e07cd810f3d558c2` | `513d3aeb03f5fa87339bfa8e183d69ff38abab70641fb3ebcaa2976eac0ff6a8` |
| `src/lcs/kernels.hip` | `d4af4ad6ad05a08da54e18f521a34aa87501ebc957382f429b246c7055837176` | `b5cfdadfeb5953e3f045c3ae4f43633d33b629a3ef79fdd0644da94a8549e5db` |
| `src/lcs/kernels.hiph` | `e553d47d8d2be708bf123fc26f8f402957f2d9e3c35f6ed3da08d514c86b793d` | `4ce070a1836eeb96eb9eadcb8e29270cef3e284fe1a79c3c62a1de3e274382d9` |
| `src/lcs/torch_hip.cpp` | `4d0a7f52700fca58ec0469b1492cc06deaece0287239adca9dfb8d85b62625fc` | `32129803de42d5799df501c814a79bdb201282bdb0e07fa2eec967d7771600c3` |
| `src/lev/kernels.hip` | `8ef6e5d089c63c17363b9dd7b6a46f2fb56984db33d34140e7291894caa8276d` | `e14484efbccf16b461f9d88db9e361a13dc60104ec2b5726fd72f3c733ef07af` |
| `src/lev/kernels.hiph` | `a96fab798aa93c0fce55a996c857582af96033ecb355fef984fd28ab0de31998` | `affeec551c910b358cb56eca81696affb9af4f0d87da6377aa3a3d91f2a33789` |
| `src/lev/torch_hip.cpp` | `da894b4ae1948abd0cee242cfa24a59a187a07acf67fbfaba6a60260fb9dd300` | `5863eee89a59ac0a4bbb80bc58985e56cec5af133e51ec52ad04fe8e83a2d711` |
| `src/mas/kernels.hip` | `6a25158acb905c372a708f6de80356b9380bb068b9ff5764e0f7747454c5dcc4` | `cb2238057d8ef586c96716a22a86376ddca73c7b41c827b834f0e9dff576c86b` |
| `src/mas/kernels.hiph` | `7b09b438de8fd340f96ef7d151ebe5ab4e1399c1e9510f625587c3ee62483d51` | `fb43d7de2f702e7c95d5b3ef72b60acdfda6a33b149205d0f60cb2d9f528266c` |
| `src/mas/torch_hip.cpp` | `cb2a3287815b4a11895ad70466f02fb75078fba559393c7e7b2808d89a184269` | `56c9c03b20d807dd483eae986ac096a90dd40524b7a2c8fcbb02551b7b6b5470` |
| `src/nw/kernels.hip` | `50cc1332e424362fecd7af8280e18fbf03450ecc013ebe7a0b0f2147028ec115` | `5211ad3058e1b4f9038240e25cf2b4087f2bef93630fc65e6c3d3c2dddf4eb1c` |
| `src/nw/kernels.hiph` | `16d4e91b4ae3c358d906a8b35ff72f231265a747ef7a4ec61e42b1c3b5d6d267` | `16d4e91b4ae3c358d906a8b35ff72f231265a747ef7a4ec61e42b1c3b5d6d267` |
| `src/nw/torch_hip.cpp` | `b840d2dfb8a7044fc8248555ecaf8f6bbd60603301c3d93e45f8618211bdbcf6` | `de12e6847b2a43982089900c79d9174cdb60fbc4c1190e09ff1d5b085176d5f3` |
| `src/nw_affine/kernels.hip` | `0ba3ec6bbd31e2d3a5ad315d2e188e14920d8f2528f235d68d8303d04a13e64a` | `a5a152987caa333c1ab7b7f5e5c7cc47549e4a7aea0dd83f7f3b2af463e8e35f` |
| `src/nw_affine/kernels.hiph` | `b89205ef56d7fdb5d20de051d934cab5d0d4b4db36a27f23c5a67907d7836991` | `b89205ef56d7fdb5d20de051d934cab5d0d4b4db36a27f23c5a67907d7836991` |
| `src/nw_affine/torch_hip.cpp` | `be4218f2b2baaaffa7427deaeb017b3e342d77a8b623a494aad2eee80910c60a` | `8b5d80f250806e8975e96bcd0662d65db519f3d09655a42775930754ec4f18d5` |
| `src/osa/kernels.hip` | `eab449dbe4f6b1eb46d43dec6002918e8e00d1c4e3ae6da87f677e5721bc998f` | `f3efdaf6eeed026aeeb17305e99f13818ac3b37f13f8d3176a3a225ef56e8c50` |
| `src/osa/kernels.hiph` | `dc1ac5212306599a2a2955dd4563f13d67f96d952f38e5fccfe47dc0797b96c9` | `88bf609caf111ba98f43511d0b25fdd7b4cc08033bc0e28f7a9c9375828b0484` |
| `src/osa/torch_hip.cpp` | `b4507b4b11303ef15aa2a0349e4858b236a439fc30d537b5db30543eed7a7ee7` | `e6733ff0f6af71c3c5bca00cda9270ddc32d6b8a85a669b162b79102d36ceba1` |
| `src/sw/kernels.hip` | `89a7c746d39ce0802026afdecd44d32bc3e78e72052a07b11857759993ec2c23` | `792385999880e975bedb92803035908c9139f4a40a16555766906515628333dd` |
| `src/sw/kernels.hiph` | `79e66b55809f6409a83161e7cb5274b768f9963525419e20863d42f1c822a4fc` | `79e66b55809f6409a83161e7cb5274b768f9963525419e20863d42f1c822a4fc` |
| `src/sw/torch_hip.cpp` | `e9e0a4ddd3a28cce8ec498f8ae1d7d291b6c601904661cae803016d07abab9b7` | `d24c2950f81ad267195621daf77bcd4a23a0b571047f9ec3eb8c4910f6cca355` |
| `src/sw_affine/kernels.hip` | `c670af3f9a42d8a354e19eee42b3b558433b7e090aecab766cd9ff4b7f51181a` | `511e11580ad739c8f613de729a284aebb07ce85c5cfca597bd5c1495db7e9d39` |
| `src/sw_affine/kernels.hiph` | `b7f2d6c3c4eb049efd673d3ad856bbc1928e47a43d0323ae67dba82e5db377d9` | `b7f2d6c3c4eb049efd673d3ad856bbc1928e47a43d0323ae67dba82e5db377d9` |
| `src/sw_affine/torch_hip.cpp` | `238f919ef46d1212503869218a892c2c7b62e6714ba4f02bdac9b0bef318d5af` | `15b81edfdedbe41328ada28fee021e195aa8b46827d0548eaacb1dc045752519` |

The HIP build and dispatch integration is release-cut glue derived from the
same committed tree and renamed to the public Orihime interfaces. It is not a
new kernel implementation. The source kernels are architecture-independent;
Meson selects one or more AMD code-object targets at build time.
