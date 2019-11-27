from __future__ import print_function, division

class VCFWriter():
    """VCF dataset."""

    def __init__(self, template_path=None, base_header=None):
        self.template_path = template_path
        self.base_header= base_header
        print('potato')

    def write_vcf(self, mat, out_path, template_path=None, base_header=None, printout=False):
        if template_path is None:
            template_path = self.template_path
        if base_header is None:
            base_header = self.base_header

        assert mat is not None
        assert out_path is not None
        assert template_path is not None
        assert base_header is not None

        NUM_ELEMS = mat.shape[0]
        new_lines = []
        pos_count = 0
        with open(template_path, 'r') as vcf_file:
            all_lines = vcf_file.readlines()
            for i, line in enumerate(all_lines):
                if line[0] == '#':
                    ## Edit Header
                    if 'CHROM' in line:
                        split = line.split('\t')
                        fmt_idx = split.index('FORMAT')
                        header_list = split[0:(fmt_idx + 1)]
                        for i_elem in range(NUM_ELEMS):
                            if i_elem is NUM_ELEMS - 1:
                                header_list.append(base_header + '{}\n'.format(i_elem))
                            else:
                                header_list.append(base_header + '{}'.format(i_elem))

                        new_line = '\t'.join(header_list)
                        new_lines.append(new_line)
                    else:
                        new_line = line
                        new_lines.append(new_line)
                else:
                    split = line.split('\t')
                    fmt_idx = split.index('GT')
                    header_list = split[0:(fmt_idx + 1)]
                    for i_elem in range(NUM_ELEMS):
                        p0, p1 = int(mat[i_elem, pos_count, 0]), int(mat[i_elem, pos_count, 1])

                        if i_elem is NUM_ELEMS - 1:
                            header_list.append('{}|{}\n'.format(p0, p1))
                        else:
                            header_list.append('{}|{}'.format(p0, p1))

                    new_line = '\t'.join(header_list)

                    new_lines.append(new_line)

                    pos_count += 1
                if printout:
                    print(new_line)

        with open(out_path, 'w') as filehandle:
            filehandle.writelines(new_lines)

    def write_map(self, map, out_path, base_header, class_mapping=['AFR','EAS','EUR'], printout=False):
        with open(out_path, 'w') as filehandle:
            for i, elem in enumerate(map):
                new_line = base_header + str(i) + '\t' + class_mapping[int(elem)] + '\n'

                if printout:
                    print(new_line)
                filehandle.write(new_line)

