import re


VALID_CHOP_REGEX = re.compile(r"^\d[A-Z0-9]{1,5}$")


def validate_chop_codes_list(chop_codes: list[str]) -> list[str]:
    """Check whether a list of CHOPs contains valid codes.

    @param chop_codes: The list of codes.
    @return: A list of valid codes only.
    """
    split_chop_codes = _split_chop_codes(chop_codes)

    valid_chop_codes = list()

    for chop_info in split_chop_codes:
        chop = chop_info[0]

        # Remove the optional dots
        chop_without_dots = chop.replace('.', '').upper()

        matches = list(re.finditer(VALID_CHOP_REGEX, chop_without_dots))

        assert(len(matches) <= 1)
        if len(matches) == 1:
            valid_chop_code = matches[0].group()
            valid_chop_code_info = ':'.join([valid_chop_code, chop_info[1], chop_info[2]])
            valid_chop_codes.append(valid_chop_code_info)

    return valid_chop_codes


def _split_chop_codes(codes_list: list[str]) -> list[list[str]]:
    """From a list of CHOP codes, which are formatted as '<code>:<side>:<date>', split them into their components.

    @param codes_list: The list of CHOP codes.
    @return: A list of the info for each code, split into strings.
    """
    return [code_with_colons.split(':') for code_with_colons in codes_list]
