import itertools
import re

from loguru import logger


def convert_unit_primap_to_primap2(unit: str, entity: str) -> str:
    """
    This function converts the emissions module style units which usually neither carry
    information about the substance nor about the time to primap2 units. The function
    also handles the exception cases where PRIMAP units do contain information about the
    substance (e.g. GtC).

    Parameters
    ----------

    unit : str
        unit to convert
    entity : str
        entity for which the conversion takes place

    Returns
    -------
    :str: converted unit
    """

    # check inputs
    if entity == "":
        logger.warning("Input entity is empty. Nothing converted.")
        return "error_" + unit + "_" + entity
    if unit == "":
        logger.warning("Input unit is empty. Nothing converted.")
        return "error_" + unit + "_" + entity

    # define exceptions
    exception_units = {
        "CO2eq": "CO2",  # convert to just CO2
        "CO₂eq": "CO2",  # convert to just CO2 (not for PRIMAP but e.g. NIRs)
        "<entity>N": "N",
        "C": "C",  # don't add variable here
    }

    # basic units
    basic_units = ["g", "t"]

    # prefixes
    si_unit_multipliers = ["", "k", "M", "G", "T", "P", "E", "Z", "Y"]

    # combines basic units with prefixes
    units_prefixes = list(itertools.product(si_unit_multipliers, basic_units))
    units_prefixes = [i[0] + i[1] for i in units_prefixes]

    # time information to add
    time_frame_str = " / yr"

    # build regexp to match the basic units with prefixes in units
    regexp_str = "("
    for this_unit in units_prefixes:
        regexp_str = regexp_str + this_unit + "|"
    regexp_str = regexp_str[0:-1] + ")"

    # remove spaces for more flexibility in input units
    unit = unit.replace(" ", "")

    # add entity and time frame to unit
    # special units will be replaced later
    unit_entity = unit + " " + entity + time_frame_str

    # check if unit has prefix
    match_pref = re.search(regexp_str, unit_entity)
    if match_pref is None:
        logger.warning("No unit prefix matched for unit. " + unit_entity)
        return "error_" + unit + "_" + entity

    # check if exception unit
    is_ex_unit = [
        re.match(regexp_str + ex_unit.replace("<entity>", entity) + "$", unit)
        is not None
        for ex_unit in exception_units
    ]

    if any(is_ex_unit):
        # we have an exception unit
        exception_unit = list(exception_units.keys())[is_ex_unit.index(True)]
        # first get the prefix and basic unit (e.g. Gt)
        pref_basic = match_pref.group(0)
        # now build the replacement
        converted_unit = (
            pref_basic + " " + exception_units[exception_unit] + time_frame_str
        )
    else:
        # standard unit
        converted_unit = unit_entity

    return converted_unit


def code_invalid_warn(code: str, message: str) -> str:
    """Log a warning and return an error code."""
    logger.warning(
        f"Category code {code!r} does not conform to specifications: {message}"
    )
    return "error_" + code


def convert_ipcc_code_primap_to_primap2(code: str) -> str:
    """
    This function converts IPCC emissions category codes from the PRIMAP-format to
    the pyCPA format which is closer to the original (but without all the dots)

    Codes that are not part of the official hierarchy (starting with IPCM or CATM)
    are not converted but returned without the 'CAT' or 'IPC' prefix unless the
    conversion is explicitly defined in the code_mapping dict.
    TODO: add other primap terminologies?

    Parameters
    ----------

    code: str
        String containing the IPCC code in PRIMAP format (IPCC1996 and IPCC2006 can be
        converted). The PRIMAP format codes consist of upper case letters and numbers
        only. These are converted back to the original formatting which includes lower
        case letters and roman numerals.

    Returns
    -------

    :str:
        string containing the category code in primap2 format

    Examples
    --------

    >>> convert_ipcc_code_primap_to_primap2("IPC1A3B34")
    '1.A.3.b.iii.4'

    """

    arabic_to_roman = {
        "1": "i",
        "2": "ii",
        "3": "iii",
        "4": "iv",
        "5": "v",
        "6": "vi",
        "7": "vii",
        "8": "viii",
        "9": "ix",
    }

    code_mapping = {
        "MAG": "M.AG",
        "MAGELV": "M.AG.ELV",
        "MBK": "M.BK",
        "MBKA": "M.BK.A",
        "MBKM": "M.BK.M",
        "MLULUCF": "M.LULUCF",
        "MMULTIOP": "M.MULTIOP",
        "M0EL": "M.0.EL",
    }

    if len(code) < 4:
        return code_invalid_warn(code, "Too short to be a PRIMAP IPCC code.")
    if code[0:3] not in ["IPC", "CAT"]:
        return code_invalid_warn(
            code, "Prefix is missing, must be one of 'IPC' or 'CAT'."
        )

    # it's an IPCC code. convert it
    # check if it's a custom code (beginning with 'M'). Currently these are the same
    # in pyCPA as in PRIMAP
    if code[3] == "M":
        code_remaining = code[3:]
        if code_remaining in code_mapping.keys():
            new_code = code_mapping[code_remaining]
            return new_code
        else:
            new_code = "M."
            code_remaining = code_remaining[1:]
    else:
        new_code = ""
        # only work with the part without 'IPC' or 'CAT'
        code_remaining = code[3:]

    # actual conversion happening here
    # first level is a digit
    if code_remaining[0].isdigit():
        new_code = new_code + code_remaining[0]
    else:
        return code_invalid_warn(code, "No digit found on first level.")
    # second level is a letter
    if len(code_remaining) > 1:
        code_remaining = code_remaining[1:]
        if code_remaining[0].isalpha():
            new_code = new_code + "." + code_remaining[0]
        else:
            return code_invalid_warn(code, "No letter found on second level.")
        # third level is a number. might be more than one char, so use regexp
        if len(code_remaining) > 1:
            code_remaining = code_remaining[1:]
            match = re.match("^[0-9]+", code_remaining)
            if match is not None:
                new_code = new_code + "." + match.group(0)
            else:
                return code_invalid_warn(code, "No number found on third level.")
            # fourth level is a letter. has to be transformed to lower case
            if len(code_remaining) > len(match.group(0)):
                code_remaining = code_remaining[len(match.group(0)) :]
                if code_remaining[0].isalpha():
                    new_code = new_code + "." + code_remaining[0].lower()
                else:
                    return code_invalid_warn(code, "No letter found on fourth level.")
                # fifth level is digit in PRIMAP1 format but roman numeral in IPCC
                # and PRIMAP2
                if len(code_remaining) > 1:
                    code_remaining = code_remaining[1:]
                    if code_remaining[0].isdigit():
                        new_code = new_code + "." + arabic_to_roman[code_remaining[0]]
                    else:
                        return code_invalid_warn(code, "No digit found on fifth level.")
                    # sixth and last level is a number.
                    if len(code_remaining) > 1:
                        code_remaining = code_remaining[1:]
                        match = re.match("^[0-9]+", code_remaining)
                        if match is not None:
                            new_code = new_code + "." + match.group(0)
                            # check if anything left
                            if not code_remaining == match.group(0):
                                return code_invalid_warn(
                                    code, "Chars left after sixth level."
                                )
                        else:
                            return code_invalid_warn(
                                code, "No number found on sixth level."
                            )

    return new_code


def convert_entity_gwp_primap_to_primap2(entity_pm1: str) -> str:
    """
    This function transforms PRIMAP1 style entity names into PRIMAP2 style variable
    names. The transformation only considers the GWP, currently the variable itself is
    unchanged.

    Currently the function uses a limited set of GWP values (defined in gwp_mapping) and
    works on a limited set of variables (defined in entities_gwp).

    Parameters
    entity: str
        entity to process

    Returns
    -------
    entity: str
        entity in PRIMAP2 format

    """

    entities_gwp = [
        "KYOTOGHG",
        "HFCS",
        "PFCS",
        "FGASES",
        "OTHERHFCS CO2EQ",
        "OTHERHFCS",
        "OTHERPFCS",
    ]

    # define the mapping of PRIMAP GWP specifications to PRIMAP2 GWP specification
    # no GWP given will be mapped to SAR
    gwp_mapping = {
        "SAR": "SARGWP100",
        "AR4": "AR4GWP100",
        "AR5": "AR5GWP100",
        "AR5CCF": "AR5CCFGWP100",  # not sure if implemented in scmdata units
    }

    # build regexp to match the GWP conversion variables
    regexp_str = "("
    for current_entity in entities_gwp:
        regexp_str = regexp_str + current_entity + "|"
    regexp_str = regexp_str[0:-1] + ")"

    # build regexp to match the GWPs
    regexp_str_gwps = "("
    for mapping in gwp_mapping.keys():
        regexp_str_gwps = regexp_str_gwps + mapping + "|"
    regexp_str_gwps = regexp_str_gwps[0:-1] + ")$"

    # check if entity in entities_gwp
    found = re.match(regexp_str, entity_pm1)
    if found is None:
        # not a basket entity which uses GWPs
        entity_pm2 = entity_pm1
    else:
        # check if GWP information present in entity
        match = re.search(regexp_str_gwps, entity_pm1)
        if match is None:
            # SAR GWPs are default in PRIMAP
            entity_pm2 = entity_pm1 + " (" + gwp_mapping["SAR"] + ")"
        else:
            gwp_out = match.group(0)
            # in this case the entity has to be replaced as well
            match = re.search(".*(?=" + gwp_out + ")", entity_pm1)
            if match.group() == "":
                logger.error(
                    "Confused: could not find entity which should be there."
                    " This indicates a bug in this function."
                )
                raise RuntimeError(
                    "Confused: could not find entity which should be there."
                    " This indicates a bug in this function."
                )
            else:
                entity_pm2 = match.group(0) + " (" + gwp_mapping[gwp_out] + ")"

    return entity_pm2
