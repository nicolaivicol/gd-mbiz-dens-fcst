# manually labeled
TREND_MANUAL = [
    1001,
    1117,
    5023, 5015, 5039, 5103,
    6059, 6005, 6019,
    8059, 8035,
    9009,
    10001,
    12089, 12031, 12111, 12107, 12011, 12051, 12123, 12086, 12069,
    13223, 13145, 13215, 13067, 13205, 13121, 13029, 13313, 13087, 13297, 13127,
    13179, 13285, 13071, 13297, 13059, 13079,
    15001,
    16001, 16055, 16071, 16083, 16075, 16019,
    17163, 17049,
    18023, 18143, 18005, 18167, 18141, 18003, 18089,
    19101, 19067,
    20163, 20091,
    21005, 21077, 21211, 21079,
    22069, 22007, 22021, 22061, 22113,
    24023, 24017,
    25009, 25023, 25027,
    26121, 26139, 26091, 26085, 26141,
    27043, 27163, 27059, 27019,
    28081, 28027, 28011, 28089, 28059, 28087, 28127,
    29025, 29083, 29125, 29225, 29131, 29221,
    30029, 30001, 30083, 30093, 30063, 30111,
    31025, 31077, 31137,
    32003,
    34039, 34001, 34021,
    35025,
    36071, 36059, 36047, 36029, 36059, 36111,
    37001, 37193, 37083, 37047, 37189, 37183, 37129, 37119, 37021,
    38071,
    39089, 39083, 39165,
    40125, 40095, 40021, 40097, 40021,
    41005, 41029,
    42055, 42011, 42017, 42073, 42095, 42069, 42011,
    44007, 44001,
    45075, 45059, 45017, 45079, 45063, 45085, 45013, 45085, 45029, 45075, 45027,
    45041, 45001, 45013, 45085,
    46071,
    47179, 47139, 47149, 48491, 47187, 47037, 47119,
    48497, 48059, 48411, 48491, 48041, 48027, 48215, 48399, 48251, 48415, 48149,
    48019, 48279, 48071, 48309, 48415, 48209, 48141, 48101, 48085, 48121, 48339,
    48439,
    51165, 51099, 51105, 51680, 51087, 51167, 51133, 51179, 51153,
    53027, 53057,
    54061, 54035, 54003, 54035,
    55045, 55015, 55139, 55133,
    56045, 56035,
]

TREND_BUT_MODEL_SAYS_NAIVE = [
    1049,
]

TREND_BUT_MODEL_SAYS_MA = [
    30095,
]

# from model
TREND_MODEL = [
    # 0.90 < w
    48339, 20091, 48209, 12086, 12069, 37119, 34039, 12097, 47187, 24033, 48491,
    48029, 21111, 47165, 12095, 45079, 12103, 36111, 47037, 45063, 37183, 40109,
    45045, 36119, 48039, 12071, 51153, 12109, 18057, 45019, 47157, 48085, 9001,
    40143, 12117, 18089, 51087, 48215, 47149,
    # 0.80 < w 0.90
    12033, 16001, 6111, 42101, 48251, 17197, 12057, 12101, 48439, 6001, 13051,
    12081, 13151, 37129, 25027, 12105, 53053, 48201, 28089, 6077, 11001, 48397,
    13077, 45013, 21067, 48027, 51179, 12111, 6019, 54061, 48157, 45083, 53057,
    42091, 48423, 24021, 47119, 9003, 48367, 13067, 34005, 34013, 34001, 6071,
    48041, 13121, 47189, 45051, 37051, 6067, 47065, 47093, 39165, 37179, 36059,
    9009, 53073, 34003, 16055, 31109, 12031, 1089, 13089, 48257, 10001, 45015,
    6029, 12083, 12011, 37097, 12015, 48183, 26139, 37021, 13247, 48113, 48303,
    39041, 22033, 12113, 51510, 39103, 8001, 29165, 31055, 13223, 41039, 39035,
    25023, 18081, 28047, 41005, 6023, 8123, 26093, 48091, 37133, 13297, 30029,
    37081, 17161, 47125, 19101, 55105, 44007, 13113, 34031, 48309, 24510, 13045,
    4025, 16021, 22005, 37001, 24017, 36005,
    # 0.70 < w 0.80
    17097, 30111, 51550, 56033, 39139, 12085, 31153, 53011, 29047, 48121, 49043,
    18141, 13059, 55009, 9011, 17037, 13215, 6061, 5007, 34029, 10005, 6107, 6065,
    42045, 46099, 42003, 13057, 45003, 48479, 12021, 37055, 34007, 12091, 47031,
    1073, 45041, 13139, 36047, 46103, 18003, 32031, 39085, 39113, 12089, 51650,
    39099, 36071, 36085, 21185, 49057, 4027, 48259, 26121, 29043, 49021, 41047,
    36081, 28071, 48329, 39089, 48167, 19049, 6073, 56025, 13013, 51177, 8035,
    28059, 37089, 44001, 48139, 4019, 13153, 6059, 42017, 51700, 19153, 42055,
    47113, 42041, 17119, 12131, 39061, 28121, 6047, 37175, 49005, 34009, 2170,
    13021, 13073, 18163, 46083, 22051, 6013, 42133, 37035, 20209, 36103, 13217,
    13097, 23031, 22017, 22105, 25017, 42029, 47163, 6039, 6099, 22063, 18097,
    29510, 25025, 6037, 27109, 27019, 17143, 49035, 32003, 12035, 36069, 42011,
    42043, 1003, 37105, 26077, 47141, 28049, 18063, 12053,
    # 0.60 < w 0.70
    55133, 24005, 6053, 39049, 51680, 37101, 28081, 28033, 17043, 48171, 42019,
    19013, 4015, 5045, 48053, 55079, 36093, 45007, 25001, 29189, 39109, 49011,
    51013, 49003, 41061, 6083, 6017, 49053, 13063, 22055, 2020, 15003, 8041,
    34025, 51590, 17099, 5125, 49051, 36087, 34023, 53025, 48441, 16027, 1097,
    53005, 20173, 8059, 26163, 51069, 22079, 15009, 6041, 13185, 39153, 42073,
    37127, 22071, 44009, 55087, 33001, 26099, 36029, 24043, 37159, 42125, 26091,
    39155, 42049, 16083, 39093, 5119, 50007, 26005, 45035, 26055, 33013, 17113,
    6079, 24027, 16005, 24013, 47147, 37067, 37063, 34037, 36067, 8077, 42077,
    39095, 30063, 12005, 36065, 6045, 27035, 33015, 40131, 48497, 16019, 37085,
    34015, 47105, 36063, 5085, 27163, 41067, 6095, 48187, 18011, 41017, 8013,
    25009, 22073, 8045, 26115, 10003, 25021, 53015, 5143, 1117, 24037, 21037,
    12023, 16017, 13313, 36061, 36055, 48473, 8037, 48231, 51059, 51800, 37087,
    39025, 2090, 48381, 13295
]

MA_MANUAL = [
    48057, 54055, 46027, 27113, 26033,
]

MA_MODEL = [
    # 0.35 < w
    18021, 31003, 29049, 51103, 47133, 18017, 40001, 5107, 54027, 48417, 35047, 55078, 21103, 37187, 20189, 1079, 54099,
    13125, 13147, 5071, 17173, 16037, 26003, 30039, 19033, 39127, 53051, 45067, 21135, 22091, 30095, 55051, 31041,
    47057, 13227, 26097, 40127, 8125, 48103, 17003, 28151, 30061, 54083, 51119, 27135, 29067, 40135, 5117, 26063, 17083,
    19157, 55031, 39001, 40133, 5137, 13307, 8027, 37033, 37013, 55119, 42005, 18173, 39143, 1085, 27065, 13321
]

NAIVE_MANUAL = [
    # naive to try with LB
    29063, 12001,
    # Zero last:
    28055,
    #
    38087, 32510, 46069, 51043, 13239,
    # CHOPPY_STEPS_SMALL_VALUES:
    30103, 38085, 46105, 20179, 30069, 20071, 15005, 38087, 51121, 13265, 21007,
    # MANUAL UNCAT
    31115, 2013, 17017, 31133, 51043, 38021, 38067, 5013, 38085, 35059, 46101,
    29111, 46039, 48443, 29137, 19085, 31165, 20191, 49019, 39037, 40085, 46003,
    38081, 46021, 47095, 38043, 46039, 31015, 20167, 53041, 46097, 20063, 31125,
    21023, 30069, 20101, 18171, 13249, 48447, 31177, 40129, 51685, 28125, 31015,
    13093, 48125, 32001, 21233, 8017, 46041, 46021, 21075, 48265, 46075, 2060,
    38037, 8057, 38039, 46045, 21217, 29139, 31171, 31163, 38095, 48151, 13243,
    21091, 17087, 20065, 19093, 54073, 8107, 13001, 21063, 17071, 48501, 8021,
    31175, 17013, 13287, 38027, 48469, 48469, 8053, 28021, 6005, 17047, 23029,
    27113, 46031, 13243, 48033, 38091, 12001, 49031, 5025, 16079, 49009,
    13093, 13201, 38073, 16041, 48179, 40149, 16025, 40153, 5011, 38025, 41051,
    26109, 21001, 46063, 13001, 29185, 13065, 31097, 49017, 30005, 13197, 12061,
    2230, 27029, 35006, 19095, 20205, 6089, 19193, 31015, 20065, 1133,
    # BAD
]

NAIVE_MODEL = [
    # model: 0.90 < w
    38047, 31183, 30011, 48269, 41055, 48443, 31009, 28055, 22023, 48301, 38029,
    13315, 38027, 31085, 36073, 48327, 55035, 27013, 46025, 53041, 38099, 26017,
    17137, 33003, 40047, 38013, 5105, 5013, 30065, 29109, 31031, 18051, 47035,
    5111, 54033, 46089, 31105, 38085, 48033, 39097, 29227, 30109, 40081, 17157,
    8051, 26111, 2110, 27137, 54039, 17183, 51121, 28035, 29081, 20101, 51085,
    31165, 21035, 47087, 42085, 32011, 20105, 48421, 26149, 12121, 17071, 37041,
    31015, 48261, 20083, 39009, 19035, 20155, 31163, 13069, 55033, 30053, 46069,
    31005, 18029, 17081, 29169, 48253, 47041, 30087, 16067, 20013, 24019, 46017,
    31139, 51159, 8025, 40017, 19131, 39115, 47167, 36107, 54105, 51141, 21123,
    18181, 18165, 51015, 48017, 21227, 8117, 31027, 36003, 51195, 54107, 26135,
    28025, 55127, 24041, 5061, 51145, 48471, 12003, 36041, 13159, 21233, 24029,
    37199, 21061, 56027, 8023, 27007, 32033, 39055, 48393, 27017, 38023, 21105,
    41003, 8043, 50025, 1105, 36099, 36009, 32019, 54101, 17089, 19165, 37091,
    31073, 21023, 51067, 17151, 48455, 46073, 29155, 13263, 20165, 42065, 42009,
    5053, 19021, 55005, 26065, 48107, 31157, 38073, 36035, 20167, 6043, 1087,
    53055, 42081, 41031, 18095, 17013, 21011, 51009, 30007, 48125, 39101, 13265,
    36109, 38003, 29205, 16007, 53067, 51063, 29001, 35035, 27049, 56013, 19079,
    47129, 51570, 19135, 46049, 31141, 2282, 29065, 48047, 30013, 20097, 13175,
    # model: 0.80 < w < 0.90
    26029, 36075, 21131, 29185, 21171, 17185, 38009, 51171, 29009, 9013, 51017,
    29085, 28077, 48163, 42107, 4023, 8093, 13027, 5067, 28075, 1057, 21219,
    5051, 21101, 37025, 20129, 24015, 21059, 29089, 17023, 20195, 13157, 55047,
    31131, 17039, 48365, 42123, 17131, 17193, 47143, 13253, 8063, 21177, 37113,
    37137, 55007, 51001, 19093, 18131, 13129, 5005, 48225, 51600, 48385, 47177,
    18113, 6007, 39147, 26089, 48451, 6113, 8014, 31051, 55039, 54067, 51750,
    51139, 48173, 48145, 8111, 47059, 28021, 40099, 20051, 46045, 39017, 35021,
    15005, 22059, 47161, 40139, 51125, 48299, 21125, 54103, 48369, 46097, 48255,
    20157, 13111, 5101, 2185, 29103, 30069, 54009, 23029, 25015, 18071, 49017,
    2275, 37151, 21083, 47185, 21087, 38007, 46101, 23015, 17087, 40105, 28153,
    53037, 20049, 28007, 4012, 38103, 46115, 42059, 21189, 21107, 36091, 38087,
    5099, 54079, 28149, 37049, 29107, 35043, 30027, 13281, 33011, 27143, 17017,
    37003, 17077, 48307, 49007, 48293, 29059, 27075, 37143, 27003, 36113, 48383,
    40149, 46039, 27161, 53019, 6097, 19163, 27081, 31029, 21199, 19011, 17147,
    48063, 42105, 37131, 21095, 17025, 51073, 51033, 51181, 40141, 26007, 13189,
    18121, 17027, 48321, 21157, 6055, 31119, 55053, 20087, 21213, 37181, 18177,
    51083, 48239, 29099, 18079, 49031, 22011, 31097, 41045, 18065, 17149, 17007,
    19107, 54059, 17159, 55129, 20045, 48335, 19133, 12129, 29173, 23019, 27153,
    13183, 18075, 47007, 19063, 27167, 27001, 42111, 13161, 5001, 21007, 16057,
    42093, 8017, 21183, 31111, 21051, 21065, 22123, 46055, 49039, 8109, 54089,
    36027, 2070, 51185, 48409, 17075, 18053, 48151, 40057, 55107, 19129, 38017,
    26053, 13219, 48347, 20025, 31047, 48375, 17203, 17135, 34035, 32027, 53013,
    46123, 35019, 38039, 38081, 51660, 27033, 1015, 41023, 30051, 51041, 48077,
    18047, 13135, 51093, 47181, 28101, 39079, 48179, 18037, 21215, 26071, 19187,
    13279, 38091, 22041, 20125, 27085, 20149, 17125, 19005, 19059, 48447, 6081,
    36095, 20135, 19137, 29115, 27111, 8039, 5075, 47027, 55067, 19037, 48427,
    31171, 47109, 41015, 40067, 42053, 13093, 46063, 48087, 39031, 2122, 38041,
    36121, 20059, 5029, 54031, 40129, 40055, 5087, 48359, 39051, 19175, 47169,
    32007, 54069, 17167, 38055, 20065, 13239, 30031, 47085, 46021, 8057, 1049,
    13309, 22127, 13257, 31115, 5043, 28039, 47033, 53033, 18183
]

TREND = list(set(TREND_MANUAL + TREND_MODEL + TREND_BUT_MODEL_SAYS_NAIVE + TREND_BUT_MODEL_SAYS_MA))
NAIVE = list(set(NAIVE_MANUAL + NAIVE_MODEL))
MA = list(set(MA_MANUAL + MA_MODEL))

NAIVE = [c for c in NAIVE if c not in TREND]
MA = [c for c in MA if c not in TREND]

if __name__ == '__main__':
    n = len(set(TREND + MA + NAIVE))
    print(f'Labeled: {n} ({n/3135*100:.1f}%)')
    print(f'- trend: {len(set(TREND))}')
    print(f'- naive: {len(set(NAIVE))}')
    print(f'-    ma: {len(set(MA))}')
