import re


"""The pattern of a valid ICD: a letter followed by 2-4 digits."""
VALID_ICD_REGEX = re.compile(r"^[A-Z]\d{2,4}$")


def validate_icd_codes_list(icd_codes: list[str]) -> list[str]:
    """Check whether a list of ICDs contains valid codes.

    @param icd_codes: The list of codes.
    @return: A list of valid codes only.
    """
    valid_icd_codes = list()

    for icd in icd_codes:
        # Remove the optional dot between the first 3 characters and the rest of the string
        icd_without_dot = icd.replace('.', '').upper()

        matches = list(re.finditer(VALID_ICD_REGEX, icd_without_dot))

        if len(matches) == 0:
            continue

        elif len(matches) == 1:
            valid_icd_code = matches[0].group()
            valid_icd_codes.append(valid_icd_code)

        else:
            raise ValueError(f'Expected one match. Got >= 1')

    return valid_icd_codes