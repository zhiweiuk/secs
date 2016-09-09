
class SymbolItem(object):
    '''
    An object of  Item is a basic element in
    a sequence.
    '''
    def __init__(self, content):
        self.content=content

    def similarTo(self,item):
        '''
        This is used to compare to another item.
        If two items have identical content,
        this returns 1; otherwise returns 0;
        :param item:
        :type item:
        :return:
        :rtype:
        '''

        if self.content!=item.content:
            return 0
        return 1

    def __str__(self):
        return self.content

    def __repr__(self):
        return str(self)


